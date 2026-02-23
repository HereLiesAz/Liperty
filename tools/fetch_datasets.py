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
    mvlrs_dir = os.path.join(data_dir, "MV-LRS")
    
    os.makedirs(lrs3_dir, exist_ok=True)
    os.makedirs(mvlrs_dir, exist_ok=True)
    
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
    
    print("\n========================================")
    print("Next Steps:")
    print("1. Obtain the datasets manually using the links above.")
    print("2. Place them in the created 'data/' directories.")
    print("3. Use 'tools/external/auto_avsr/preprocessing' scripts to crop/process video.")
    print("4. Use 'tools/create_trainable_model.py' to generate a LoRA-ready model.")
    print("========================================")

if __name__ == "__main__":
    setup_datasets()
