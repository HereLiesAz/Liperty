import tensorflow as tf
import os

def create_dummy_model():
    # Define input shape: (Batch=1, Time=50, Height=88, Width=88, Channel=1)
    input_layer = tf.keras.layers.Input(shape=(50, 88, 88, 1), batch_size=1)

    # Simple transformation to reach output shape (1, 50, 40)
    # 1. Average pooling over spatial dimensions to get (1, 50, 1, 1, 1) -> (1, 50, 1)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(input_layer)

    # Now x is (1, 50, 1)
    # 2. Dense layer to project to 40 classes
    output_layer = tf.keras.layers.Dense(40, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_path = "app/src/main/assets/vsr_model.tflite"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    create_dummy_model()
