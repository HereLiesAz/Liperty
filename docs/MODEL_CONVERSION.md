# Visual Speech Recognition (VSR) Model Conversion Guide

This guide outlines the process of converting PyTorch-based VSR models (like VALLR) into TensorFlow Lite format for use in the Android application.

## Prerequisites

1.  **Python Environment**: Ensure you have Python 3.10+ installed.
2.  **Dependencies**: Install the following Python libraries:
    ```bash
    pip install torch onnx onnx-tf tensorflow
    ```
3.  **VALLR Model**: Download the pre-trained VALLR model checkpoint (`.pth` or `.ckpt`).
    -   *Source*: [VALLR Repository](https://github.com/MarshallT-99/VALLR) (or check releases for pre-trained weights).

## Conversion Steps

We provide a template script `tools/convert_vallr.py` to facilitate the conversion.

1.  **Prepare the Script**:
    -   Open `tools/convert_vallr.py`.
    -   Import the actual VALLR model definition (you may need to clone the VALLR repo and add it to `PYTHONPATH`).
    -   Update the `pytorch_model_path` variable to point to your downloaded checkpoint.
    -   Update the input shape if necessary (Default: `(1, 50, 88, 88, 1)`).

2.  **Run the Conversion**:
    ```bash
    python tools/convert_vallr.py
    ```
    This script performs the following:
    -   Loads the PyTorch model.
    -   Exports it to ONNX format.
    -   Converts the ONNX model to TensorFlow SavedModel.
    -   Converts the TensorFlow SavedModel to TFLite (`.tflite`).

3.  **Verify the Output**:
    -   The script generates `app/src/main/assets/vsr_model.tflite`.
    -   Ensure the file size is reasonable (e.g., > 10MB depending on model architecture).

## Using a Dummy Model (For Development)

If you do not have the VALLR model or need to test the pipeline without it, you can generate a dummy TFLite model.

1.  **Run the Dummy Generator**:
    ```bash
    python tools/create_dummy_model.py
    ```
    This creates a placeholder `vsr_model.tflite` with the correct input/output shapes but random weights. This allows the app to compile and run the inference pipeline without crashing.

## Troubleshooting

-   **ONNX Export Errors**: Ensure the model is in `eval()` mode and doesn't use operations not supported by ONNX.
-   **TFLite Conversion Errors**: Check if custom ops are used. You might need to enable `SELECT_TF_OPS`.
-   **Input Shape Mismatch**: Verify that the input tensor shape in `tools/create_dummy_model.py` matches what the Android app expects in `VSRInference.kt`.
