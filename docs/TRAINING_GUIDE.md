# VSR Training & Fine-Tuning Guide

This guide describes the multi-dataset strategy for training Liperty's Visual Speech Recognition models.

## Training Environments

### 1. Cloud Training (Recommended)
For most users, we recommend using **Google Colab** to leverage high-performance GPUs (T4/A100) for free or at low cost.
- **Notebook**: [liperty_vsr_training.ipynb](file:///home/az/StudioProjects/Liperty/tools/liperty_vsr_training.ipynb)
- **Features**: Automated setup, Google Drive integration for checkpoints, and pre-configured dependency installation.

### 2. Local Training
Requires a Linux environment with an NVIDIA GPU and CUDA 11.8+. 
- Use `./setup_libs.sh` to initialize the environment.

## Dataset Strategy

### 1. Pre-training: Oxford LRS3, LRS2 & LRW-1
*   **Goal**: Learn robust, pose-invariant visual spatiotemporal features.
*   **Method**: 
    - Use **LRW-1** for high-density word-level pre-training.
    - Use **LRS3/LRS2** for sentence-level context.
    - Use **LRS2-2Mix** to improve robustness in multi-speaker "cocktail party" environments.

### 2. Specialized Robustness: MIRACL-VC1 (RGB-D) & VVAD-LRS3
*   **Goal**: Low-light accuracy and intelligent power management.
*   **Method**: 
    - Fine-tune on **MIRACL-VC1** for depth-augmented VSR.
    - Train a lightweight **Visual Voice Activity Detection (VVAD)** model on **VVAD-LRS3** to gate the main VSR engine, saving battery during silence.

### 3. Verification & Unit Testing: GRID (via HF)
*   **Goal**: Ensure phoneme-to-index mapping is accurate.
*   **Method**: Test on clean, frontal lab data. If the model fails on GRID, the pipeline architecture or vocabulary mapping is likely broken.

## Workflow

1.  **Environment Setup**:
    ```bash
    ./setup_libs.sh
    pip install datasets transformers
    ```

2.  **Fetch Data**:
    ```bash
    python tools/fetch_datasets.py
    ```
    *Follow manual instructions for LRS2/LRS3 access.*

3.  **Preprocessing**:
    Use `tools/external/auto_avsr/preprocessing` to:
    - Detect landmarks.
    - Crop mouth ROIs (88x88).
    - Normalize luminance.

4.  **Training (LoRA)**:
    Use `tools/create_trainable_model.py` to initialize a VALLR model with LoRA (Low-Rank Adaptation) layers. This allows for fast fine-tuning on a phone without retraining the base LLM decoder.

5.  **Conversion**:
    ```bash
    python tools/convert_vallr.py
    ```
    Converts the trained PyTorch model to TFLite for deployment in `app/src/main/assets/vsr_model.tflite`.
