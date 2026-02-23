import tensorflow as tf
import os

def create_dummy_model():
    # Define input shape: (Batch=1, Time=50, Height=88, Width=88, Channel=1)
    # This shape is based on common VSR model inputs like LipNet/VALLR (as per RESEARCH.md)
    input_layer = tf.keras.layers.Input(shape=(50, 88, 88, 1), batch_size=1, name="input_video")

    # Simple transformation to reach output shape (1, 50, 40)
    # 1. TimeDistributed Conv2D to process frames
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))(input_layer)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)

    # Flatten spatial dimensions: (1, 50, H, W, C) -> (1, 50, Features)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    # Recurrent layers to model temporal dependencies
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)

    # Output layer: Project to 40 character classes (including CTC blank)
    output_layer = tf.keras.layers.Dense(40, activation='softmax', name="ctc_output")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Convert to TFLite with flex ops if needed, or standard
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable resource variables and Select TF ops to support LSTMs if standard conversion fails
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    output_path = "app/src/main/assets/vsr_model.tflite"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    create_dummy_model()
