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
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # 3. Define Training Step (The Signature)
    @tf.function(input_signature=[
        tf.TensorSpec([None, 50, 88, 88, 1], tf.float32),  # Input Video
        tf.TensorSpec([None, 50, 40], tf.float32)          # Target Labels (One-Hot)
    ])
    def train(video_input, target_labels):
        with tf.GradientTape() as tape:
            predictions = model(video_input, training=True)
            loss = loss_fn(target_labels, predictions)
            
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
