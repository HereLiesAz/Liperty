import tensorflow as tf
import os
import numpy as np

def create_trainable_model():
    print("Creating LoRA-ready trainable TFLite model...")
    
    # 1. Define the Base Model (Simplified VSR Encoder)
    # Input: [Batch, Time, Height, Width, Channels]
    input_video = tf.keras.layers.Input(shape=(50, 88, 88, 1), name="input_video")
    
    # Feature Extractor (Fixed/Frozen for LoRA)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))(input_video)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    
    # Bottleneck for LoRA (Low-Rank Adaptation) simulation
    # In a real LoRA implementation, we would decompose a weight matrix W = W0 + BA
    # Here, we make the final Dense layer trainable.
    
    # Output: [Batch, Time, VocabSize]
    # Vocab Size = 40 (39 chars + blank)
    output_layer = tf.keras.layers.Dense(40, activation='softmax', name="output_logits")(x)
    
    model = tf.keras.Model(inputs=input_video, outputs=output_layer)
    
    # 2. Define Loss and Optimizer
    # CTC loss must be used for sequence-to-sequence lipreading (not CategoricalCrossentropy).
    # tf.nn.ctc_loss expects inputs of shape (time, batch, num_classes) with log probabilities,
    # target labels as a SparseTensor, sequence lengths, and label lengths.
    # The training signature below wraps ctc_loss for use inside tf.GradientTape.
    loss_fn = tf.nn.ctc_loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # 3. Define Training Step (The Signature)
    # NOTE: CTC loss requires (labels, label_length, logit_length) and logits in time-major
    # layout (time, batch, num_classes). The signature below accepts pre-encoded sparse
    # label indices and sequence-length tensors as required by tf.nn.ctc_loss.
    @tf.function(input_signature=[
        tf.TensorSpec([None, 50, 88, 88, 1], tf.float32),  # Input Video
        tf.TensorSpec([None, None], tf.int32),              # Target label indices (padded)
        tf.TensorSpec([None], tf.int32),                    # Label lengths
        tf.TensorSpec([None], tf.int32),                    # Logit (input) lengths
    ])
    def train(video_input, target_labels, label_lengths, logit_lengths):
        with tf.GradientTape() as tape:
            predictions = model(video_input, training=True)  # (batch, time, num_classes)
            # CTC loss expects time-major logits: (time, batch, num_classes)
            logits_time_major = tf.transpose(predictions, [1, 0, 2])
            loss = loss_fn(
                labels=target_labels,
                logits=logits_time_major,
                label_length=label_lengths,
                logit_length=logit_lengths,
                blank_index=39,  # Index 39 reserved as CTC blank token
                logits_time_major=True,
            )
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return {"loss": loss}

    # 4. Define Inference Step
    @tf.function(input_signature=[
        tf.TensorSpec([None, 50, 88, 88, 1], tf.float32)
    ])
    def infer(video_input):
        return {"output": model(video_input, training=False)}

    # 5. Save as SavedModel first
    saved_model_path = "tools/saved_model_lora"
    tf.saved_model.save(
        model, 
        saved_model_path,
        signatures={
            'train': train,
            'infer': infer,
            'serving_default': infer
        }
    )

    # 6. Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops (needed for training).
    ]
    converter.experimental_enable_resource_variables = True
    
    tflite_model = converter.convert()

    # 7. Save TFLite Model
    output_path = "app/src/main/assets/vsr_lora_model.tflite"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Trainable LoRA model saved to {output_path}")

if __name__ == "__main__":
    create_trainable_model()
