import subprocess
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="SubZero Smart Master Pipeline")
    parser.add_argument("input", help="Path to input video or frame folder")
    parser.add_argument("--preview", action="store_true", help="Run in preview mode")
    args = parser.parse_args()

    input_target = args.input

    if not os.path.exists(input_target):
        print(f"ERROR: Input path '{input_target}' does not exist.")
        sys.exit(1)
    
    # ---- Smart output logic ----
    if os.path.isdir(input_target):
        # Input is an image sequence folder.
        output_dir = input_target
    else:
        # Input is a video file. Create a dedicated output directory next to it.
        parent_dir = os.path.dirname(input_target)
        filename = os.path.splitext(os.path.basename(input_target))[0]
        output_dir = os.path.join(parent_dir, f"{filename}_subzero_output")
        os.makedirs(output_dir, exist_ok=True)
        
    extra_flags = ["--preview"] if args.preview else []

    # ---- NEW NORMALS SUBFOLDER LOGIC ----
    # Explicitly force a clean "normals" folder inside the target output directory
    normals_out_dir = os.path.join(output_dir, "normals")
    os.makedirs(normals_out_dir, exist_ok=True)

    # Launch normals script
    print("\n" + "="*60)
    print(f"STARTING: Surface Normals extraction -> Output: {normals_out_dir}")
    print("="*60)
    
    # ---- FIXED ARGUMENT FLAGS HERE ----
    # Prepend '--input' and '--output' so 'subzero_normals.py' reads them perfectly
    normals_cmd = [
        sys.executable, "subzero_normals.py", 
        "--input", input_target, 
        "--output", normals_out_dir
    ] + extra_flags
    subprocess.run(normals_cmd, check=True)

    print("\n" + "="*60)
    print(f"ALL PHASES COMPLETE! Maps saved neatly inside: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()