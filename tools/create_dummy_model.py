import tensorflow as tf
import os

def create_dummy_model():
    # Define input shape: (Batch=1, Time=50, Height=96, Width=96, Channel=1)
    # This shape is based on common VSR model inputs like LipNet/VALLR (as per RESEARCH.md)
    input_layer = tf.keras.layers.Input(shape=(50, 96, 96, 1), batch_size=1, name="input_video")

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
    # Using a kernel initializer that favors index 0 (blank) to avoid "VG" bias
    output_layer = tf.keras.layers.Dense(
        40,
        activation='softmax',
        name="ctc_output",
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer=tf.keras.initializers.Constant(value=-10.0) # Low bias for tokens
    )(x)

    # Set bias for index 0 (blank) to be high
    # (Note: This is hard to do in a simple Keras functional API setup without a custom layer,
    # but we can try to adjust the bias of the dense layer after creation)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Manually adjust bias to favor blank (index 0) to prevent constant "VG" output
    weights = model.get_layer("ctc_output").get_weights()
    biases = weights[1]
    biases[0] = 10.0 # High bias for blank
    model.get_layer("ctc_output").set_weights([weights[0], biases])

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
