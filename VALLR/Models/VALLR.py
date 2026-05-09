import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import VideoMAEConfig, Wav2Vec2Config, Wav2Vec2ForCTC, VideoMAEModel

# HuggingFace ID of the VideoMAE-base checkpoint pretrained on Kinetics-400
# (~240 hrs of diverse video). Loading these weights is the difference between
# the encoder having a useful spatial/temporal prior vs being random noise --
# which previously caused our CTC head to collapse to its argmax-over-prior
# instead of learning. ~88M params, ~360 MB download on first use.
_VIDEOMAE_PRETRAINED = "MCG-NJU/videomae-base"


class VALLR(nn.Module):
    def __init__(self, videomae_config: VideoMAEConfig, wav2vec_config: Wav2Vec2Config, adapter_dim: int,
                 pretrained_videomae: str | None = _VIDEOMAE_PRETRAINED):
        super(VALLR, self).__init__()

        # Initialize the VideoMAE encoder. Default loads pretrained weights
        # from HuggingFace; pass pretrained_videomae=None to fall back to
        # random init (config-only) -- only useful for unit tests.
        if pretrained_videomae is None:
            self.videomae = VideoMAEModel(videomae_config)
        else:
            # `from_pretrained` uses the supplied config to override the
            # pretrained config's input shape (num_frames, image_size, etc.)
            # while keeping the trained weight tensors. ignore_mismatched_sizes
            # lets us swap the default 224x224x16 input for our 88x88x16 etc.
            self.videomae = VideoMAEModel.from_pretrained(
                pretrained_videomae,
                config=videomae_config,
                ignore_mismatched_sizes=True,
            )

        # VideoMAE feature size
        videomae_feature_size = videomae_config.hidden_size  # Typically 768 for VideoMAE

        # Downsample the time dimension using multiple Conv1D and Pooling layers
        self.downsampling = nn.Sequential(
            nn.Conv1d(in_channels=videomae_feature_size, out_channels=adapter_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),

            nn.Conv1d(in_channels=adapter_dim, out_channels=adapter_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),

            nn.Conv1d(in_channels=adapter_dim, out_channels=adapter_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=adapter_dim, out_channels=adapter_dim, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5, stride=8)  # Adjust kernel size and stride to control final length
        )

        # Adapter: Adjust features from the downsampled time dimension to match Wav2Vec2 hidden size
        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim, wav2vec_config.hidden_size),  # Map to Wav2Vec2 hidden size
            nn.ReLU(),
        )

        # Initialize Wav2Vec2 CTC model
        self.wav2vec2 = Wav2Vec2ForCTC(wav2vec_config)  # No pretraining

        # Extract the final CTC classification head from Wav2Vec2
        self.ctc_head = self.wav2vec2.lm_head  # The CTC head for phoneme classification

        # **Freeze Wav2Vec2 except for CTC head**
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        for param in self.ctc_head.parameters():
            param.requires_grad = True

        # Apply weight init ONLY to the freshly-built submodules. We must
        # NOT walk the whole self with self.apply -- that would overwrite
        # every Linear layer inside the VideoMAE encoder we just loaded
        # from pretrained weights, undoing the load and putting us right
        # back into the prior-collapse failure mode.
        self.downsampling.apply(self._init_weights)
        self.adapter.apply(self._init_weights)
        self.ctc_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, nonlinearity='relu')  # Kaiming initialization for Conv1D
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization for Linear layers
            if module.bias is not None:
                init.zeros_(module.bias)

    def forward(self, video_inputs):
        # Step 1: Extract features from VideoMAE
        videomae_outputs = self.videomae(video_inputs)
        video_features = videomae_outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        print(f"Video features shape: {video_features.shape}")

        # Step 2: Downsample the time dimension using Conv1D + Pooling
        # Convert (batch_size, seq_len, feature_dim) -> (batch_size, feature_dim, seq_len)
        video_features = video_features.permute(0, 2, 1)

        # Apply Conv1D and Pooling to reduce the sequence length
        downsampled_features = self.downsampling(video_features)  # (batch_size, adapter_dim, reduced_seq_len)
        print(f"Downsampled features shape: {downsampled_features.shape}")

        # Convert back to (batch_size, reduced_seq_len, adapter_dim)
        downsampled_features = downsampled_features.permute(0, 2, 1)

        # Step 3: Adapt the features to match Wav2Vec2 hidden size
        adapted_features = self.adapter(downsampled_features)  # (batch_size, reduced_seq_len, wav2vec_hidden_size)

        # Step 4: Compute logits using the extracted CTC head
        logits = self.ctc_head(adapted_features)  # Shape: (batch_size, reduced_seq_len, num_phonemes)

        # Return raw logits
        return logits, adapted_features

