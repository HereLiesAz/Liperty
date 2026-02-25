#!/usr/bin/env python3
"""
Convert VALLR PyTorch model to TensorFlow Lite format.
Supports both V1 (VideoMAE+Wav2Vec2) and V2 (ML_VALLR) architectures.
"""

import torch
import tensorflow as tf
import numpy as np
import argparse
import os
from Models.ML_VALLR import VALLR as ML_VALLR
from Models.VALLR import VALLR
from transformers import VideoMAEConfig, Wav2Vec2Config
from config import get_vocab


class PyTorchToTFLiteConverter:
    """Handles conversion from PyTorch to TFLite"""
    
    def __init__(self, model_path, version, output_path, quantize=False):
        self.model_path = model_path
        self.version = version
        self.output_path = output_path
        self.quantize = quantize
        self.device = torch.device("cpu")  # Use CPU for conversion
        self.vocab = get_vocab()
        
    def load_pytorch_model(self):
        """Load the PyTorch model from checkpoint"""
        print(f"Loading PyTorch model (version {self.version})...")
        
        if self.version == "V1":
            videomae_config = VideoMAEConfig()
            wav2vec_config = Wav2Vec2Config()
            wav2vec_config.vocab_size = len(self.vocab)
            
            model = VALLR(
                videomae_config=videomae_config,
                wav2vec_config=wav2vec_config,
                adapter_dim=256,
            )
        elif self.version == "V2":
            model = ML_VALLR(
                adapter_dim=256,
                num_classes=len(self.vocab)
            )
        else:
            raise ValueError(f"Unknown version: {self.version}")
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def export_to_onnx(self, model, onnx_path):
        """Export PyTorch model to ONNX format"""
        print("Exporting to ONNX...")
        
        # Create dummy input (batch_size=1, channels=3, time=16, height=224, width=224)
        if self.version == "V1":
            # VideoMAE expects (batch, num_frames, channels, height, width)
            dummy_input = torch.randn(1, 16, 3, 224, 224)
        else:
            # ML_VALLR expects (batch, channels, time, height, width)
            dummy_input = torch.randn(1, 3, 16, 224, 224)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['video_input'],
            output_names=['logits', 'features'],
            dynamic_axes={
                'video_input': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'features': {0: 'batch_size'}
            }
        )
        print(f"ONNX model saved to {onnx_path}")
        return dummy_input
    
    def onnx_to_tensorflow(self, onnx_path, tf_saved_model_path):
        """Convert ONNX model to TensorFlow SavedModel"""
        print("Converting ONNX to TensorFlow SavedModel...")
        
        try:
            import onnx
            from onnx_tf.backend import prepare
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(tf_saved_model_path)
            
            print(f"TensorFlow SavedModel saved to {tf_saved_model_path}")
            return True
        except ImportError:
            print("ERROR: onnx-tf not installed. Install with: pip install onnx onnx-tf")
            return False
        except Exception as e:
            print(f"ERROR during ONNX->TF conversion: {e}")
            return False
    
    def tensorflow_to_tflite(self, tf_saved_model_path, tflite_path):
        """Convert TensorFlow SavedModel to TFLite"""
        print("Converting TensorFlow SavedModel to TFLite...")
        
        # Load the SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
        
        # Apply optimizations if requested
        if self.quantize:
            print("Applying post-training quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # For full integer quantization (optional)
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # Enable experimental features if needed
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS      # TF ops (fallback)
        ]
        
        # Convert
        try:
            tflite_model = converter.convert()
            
            # Save to file
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TFLite model saved to {tflite_path}")
            print(f"Model size: {os.path.getsize(tflite_path) / (1024*1024):.2f} MB")
            return True
        except Exception as e:
            print(f"ERROR during TFLite conversion: {e}")
            print("\nTrying with reduced ops set...")
            # Try again with full TF op support
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.allow_custom_ops = True
            
            try:
                tflite_model = converter.convert()
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                print(f"TFLite model saved to {tflite_path} (with custom ops)")
                return True
            except Exception as e2:
                print(f"ERROR: Failed to convert: {e2}")
                return False
    
    def convert_direct_pytorch_to_tflite(self, model, dummy_input):
        """
        Direct conversion from PyTorch to TFLite using traced model
        This is an alternative approach if ONNX path fails
        """
        print("\nAttempting direct PyTorch->TFLite conversion...")
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(model, dummy_input)
            traced_path = self.output_path.replace('.tflite', '_traced.pt')
            traced_model.save(traced_path)
            print(f"Traced model saved to {traced_path}")
            
            # You would need a custom converter here
            print("Note: Direct PyTorch->TFLite conversion requires additional tools.")
            print("Consider using: https://github.com/alibaba/TinyNeuralNetwork")
            return False
            
        except Exception as e:
            print(f"ERROR in direct conversion: {e}")
            return False
    
    def convert(self):
        """Main conversion pipeline"""
        print("="*60)
        print("VALLR PyTorch to TFLite Conversion")
        print("="*60)
        
        # Create output directory
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load PyTorch model
        model = self.load_pytorch_model()
        
        # Step 2: Export to ONNX
        onnx_path = self.output_path.replace('.tflite', '.onnx')
        dummy_input = self.export_to_onnx(model, onnx_path)
        
        # Step 3: Convert ONNX to TensorFlow SavedModel
        tf_saved_model_path = self.output_path.replace('.tflite', '_savedmodel')
        success = self.onnx_to_tensorflow(onnx_path, tf_saved_model_path)
        
        if not success:
            print("\nONNX->TF path failed. Trying alternative approaches...")
            self.convert_direct_pytorch_to_tflite(model, dummy_input)
            return
        
        # Step 4: Convert TensorFlow SavedModel to TFLite
        success = self.tensorflow_to_tflite(tf_saved_model_path, self.output_path)
        
        if success:
            print("\n" + "="*60)
            print("✓ Conversion completed successfully!")
            print("="*60)
            print(f"Output: {self.output_path}")
            
            # Print usage instructions
            self.print_usage_instructions()
        else:
            print("\n" + "="*60)
            print("✗ Conversion failed")
            print("="*60)
    
    def print_usage_instructions(self):
        """Print instructions for using the TFLite model"""
        print("\nUsage Instructions:")
        print("-" * 60)
        print("Python:")
        print("""
import tensorflow as tf
import numpy as np

# Load the model
interpreter = tf.lite.Interpreter(model_path="{}")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input (shape: [1, 3, 16, 224, 224] or [1, 16, 3, 224, 224])
video_input = np.random.randn(*input_details[0]['shape']).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], video_input)
interpreter.invoke()

# Get outputs
logits = interpreter.get_tensor(output_details[0]['index'])
features = interpreter.get_tensor(output_details[1]['index'])

print(f"Logits shape: {{logits.shape}}")
print(f"Features shape: {{features.shape}}")
""".format(self.output_path))
        
        print("\nAndroid (Java/Kotlin):")
        print("""
// Add to build.gradle
implementation 'org.tensorflow:tensorflow-lite:2.13.0'

// Load and run model
Interpreter tflite = new Interpreter(loadModelFile());
float[][][][] input = new float[1][16][224][224][3];
float[][][] logits = new float[1][OUTPUT_SEQ_LEN][NUM_CLASSES];
tflite.run(input, logits);
""")


def main():
    parser = argparse.ArgumentParser(description='Convert VALLR PyTorch model to TFLite')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the PyTorch model checkpoint (.pth)')
    parser.add_argument('--version', type=str, choices=['V1', 'V2'], required=True,
                        help='Model version: V1 (VideoMAE+Wav2Vec2) or V2 (ML_VALLR)')
    parser.add_argument('--output', type=str, default='vallr_model.tflite',
                        help='Output path for the TFLite model')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply post-training quantization to reduce model size')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        return
    
    # Run conversion
    converter = PyTorchToTFLiteConverter(
        model_path=args.model_path,
        version=args.version,
        output_path=args.output,
        quantize=args.quantize
    )
    
    converter.convert()


if __name__ == "__main__":
    main()
