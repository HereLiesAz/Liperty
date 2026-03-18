import os
import subprocess
import sys

def setup_datasets():
    print("========================================")
    print("Liperty Dataset Setup Helper")
    print("========================================")
    
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    lrs3_dir = os.path.join(data_dir, "LRS3")
    lrs2_dir = os.path.join(data_dir, "LRS2")
    lrs2_2mix_dir = os.path.join(data_dir, "LRS2-2Mix")
    mvlrs_dir = os.path.join(data_dir, "MV-LRS")
    lrw_dir = os.path.join(data_dir, "LRW")
    miracl_dir = os.path.join(data_dir, "MIRACL-VC1")
    grid_dir = os.path.join(data_dir, "GRID")
    vvad_dir = os.path.join(data_dir, "VVAD-LRS3")
    
    os.makedirs(lrs3_dir, exist_ok=True)
    os.makedirs(lrs2_dir, exist_ok=True)
    os.makedirs(lrs2_2mix_dir, exist_ok=True)
    os.makedirs(mvlrs_dir, exist_ok=True)
    os.makedirs(lrw_dir, exist_ok=True)
    os.makedirs(miracl_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(vvad_dir, exist_ok=True)
    
    print(f"[+] Created data directories at: {data_dir}")
    
    # 1. Auto-AVSR Integration
    print("\n[1] Fetching Auto-AVSR (Recommended for LRS3/MV-LRS preprocessing)...")
    external_tools = os.path.join(project_root, "tools", "external")
    os.makedirs(external_tools, exist_ok=True)
    auto_avsr_path = os.path.join(external_tools, "auto_avsr")
    
    if not os.path.exists(auto_avsr_path):
        try:
            subprocess.run(["git", "clone", "https://github.com/mpc001/auto_avsr.git", auto_avsr_path], check=True)
            print("[+] Cloned Auto-AVSR successfully.")
            print(f"    Check {auto_avsr_path}/preprocessing for data preparation scripts.")
        except Exception as e:
            print(f"[!] Failed to clone Auto-AVSR: {e}")
            print("    Please manually clone: https://github.com/mpc001/auto_avsr")
    else:
        print("[*] Auto-AVSR already exists.")

    # 2. LRS3-TED Instructions
    print("\n[2] LRS3-TED Dataset Instructions")
    print("    URL: https://mmai.io/datasets/lip_reading/")
    print("    Status: Requires form submission/request.")
    print("    Action: Go to the URL, request access, and download the 'LRS3' archive.")
    print(f"    Extract to: {lrs3_dir}")
    
    # 3. MV-LRS Instructions
    print("\n[3] MV-LRS (Multi-View) Dataset Instructions")
    print("    URL: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/mvlrs.html")
    print("    Status: Requires academic license agreement.")
    print("    Action: Email the maintainers (see URL) to request access.")
    print(f"    Extract to: {mvlrs_dir}")
    
    # 4. Oxford LRS2 Instructions
    print("\n[4] Oxford LRS2 Dataset Instructions")
    print("    URL: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html")
    print("    Status: Requires academic license agreement (BBC R&D).")
    print("    Action: Go to the URL, follow the registration process.")
    print(f"    Extract to: {lrs2_dir}")

    # 5. MIRACL-VC1 Instructions
    print("\n[5] MIRACL-VC1 (RGB-D Dataset) Instructions")
    print("    URL 1 (Official): https://abenhamadou.github.io/miraclvc1/")
    print("    URL 2 (Kaggle): https://www.kaggle.com/datasets/apoorvwatsky/miraclvc1")
    print("    Action: Go to the URL above, download the RGB-D dataset.")
    print(f"    Extract to: {miracl_dir}")

    # 5a. VVAD-LRS3 Instructions (Kaggle)
    print("\n[5a] VVAD-LRS3 (Visual VAD Gating)")
    print("    URL: https://www.kaggle.com/datasets/adrianlubitz/vvadlrs3")
    print("    Action: Download for training power-efficient voice activity detection.")
    print(f"    Extract to: {vvad_dir}")

    # 6. GRID (Hugging Face) Instructions
    print("\n[6] GRID Dataset (Hugging Face Subset)")
    print("    Package: datasets (pip install datasets)")
    print("    Command: from datasets import load_dataset; ds = load_dataset('wissemkarous/lipreading')")
    print(f"    Local Backup Path: {grid_dir}")
    
    print("\n========================================")
    print("Next Steps:")
    print("1. Obtain the datasets manually using the links above.")
    print("2. Place them in the created 'data/' directories.")
    print("3. Use 'tools/external/auto_avsr/preprocessing' scripts to crop/process video.")
    print("4. Use 'tools/create_trainable_model.py' to generate a LoRA-ready model.")
    print("========================================")

if __name__ == "__main__":
    setup_datasets()
