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

We provide a stub script `tools/convert_vallr.py` to illustrate the conversion steps. For real VALLR V2 model conversions, use `VALLR/convert_to_tflite.py`, which contains the complete model definition imports and correct export logic.

**Input shape note**: The legacy dummy model uses shape `(1, 50, 88, 88, 1)` (batch, frames, H, W, channels). The actual VALLR V2 model uses shape `(1, 3, 16, 224, 224)` (batch, channels, frames, H, W) following standard PyTorch channel-first convention. Ensure the script and the Android app's `VSRInference.kt` are configured for the same shape.

1.  **Prepare the Script**:
    -   For real conversions, open `VALLR/convert_to_tflite.py`.
    -   For a quick reference stub only, open `tools/convert_vallr.py` (note: this is a stub and does not contain full model definitions).
    -   Update the `pytorch_model_path` variable to point to your downloaded checkpoint.
    -   Confirm the input shape matches your model variant (legacy dummy: `(1, 50, 88, 88, 1)`; VALLR V2: `(1, 3, 16, 224, 224)`).

2.  **Run the Conversion**:
    ```bash
    # For real VALLR V2 conversions (recommended):
    python VALLR/convert_to_tflite.py

    # For stub/reference only:
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
-   **Input Shape Mismatch**: Verify that the input tensor shape used during conversion matches what the Android app expects in `VSRInference.kt`. The legacy dummy model uses `(1, 50, 88, 88, 1)`; the real VALLR V2 model uses `(1, 3, 16, 224, 224)`. These are not interchangeable without updating both the conversion script and the app's preprocessing logic.
