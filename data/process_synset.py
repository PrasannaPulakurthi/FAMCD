import os
import shutil
import pandas as pd

def copy_images(csv_path, image_root, target_root):
    df = pd.read_csv(csv_path, header=None)  # Only one column, no header

    for rel_path in df[0]:
        # rel_path example: "Cycles\\0_Geschwindigkeit20\\0_cycles.png"
        parts = rel_path.strip().split("\\")  # or .split("/") on Unix
        class_folder = parts[1]               # e.g., "0_Geschwindigkeit20"
        class_id = class_folder.split("_")[0].zfill(5)
        filename = parts[-1]

        src = os.path.join(image_root, class_folder, filename)
        dst_dir = os.path.join(target_root, class_id)
        dst = os.path.join(dst_dir, filename)

        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"Missing image: {src}")

# --- Paths ---
base_dir = "SynsetSignsetGermany"
csv_train = os.path.join(base_dir, "CsvFiles/cyclesGTSRB_train.csv")
csv_val   = os.path.join(base_dir, "CsvFiles/cyclesGTSRB_val.csv")
image_dir = os.path.join(base_dir, "Cycles")

output_train = "./SYNSET/train"
output_test  = "./SYNSET/test"

# --- Execute ---
copy_images(csv_train, image_dir, output_train)
copy_images(csv_val, image_dir, output_test)

print("Done organizing SYNSET cycles images.")
