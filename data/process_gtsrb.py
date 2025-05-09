import os
import pandas as pd
import shutil

# --- Paths ---
image_dir = "./GTSRB/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
csv_file = "./GT-final_test.csv"
output_dir = "./GTSRB/test"

# --- Create output dir ---
os.makedirs(output_dir, exist_ok=True)

# --- Load CSV with semicolon delimiter ---
df = pd.read_csv(csv_file, sep=';')

# --- Move test images into class folders ---
for _, row in df.iterrows():
    filename = row['Filename']
    class_id = str(row['ClassId']).zfill(5)  # ensure folder names are zero-padded
    src_path = os.path.join(image_dir, filename)
    class_dir = os.path.join(output_dir, class_id)
    os.makedirs(class_dir, exist_ok=True)
    dst_path = os.path.join(class_dir, filename)
    shutil.copyfile(src_path, dst_path)

print("Test images organized into class-specific folders.")

# --- Paths ---
image_dir = "./GTSRB/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
output_dir = "./GTSRB/train"

os.makedirs(output_dir, exist_ok=True)

# --- Loop over all train subfolders in image_dir ---
for folder_name in os.listdir(image_dir):
    src = os.path.join(image_dir, folder_name)
    dst = os.path.join(output_dir, folder_name)
    
    if os.path.isdir(src):
        shutil.move(src, dst)

print(f"All class folders moved from {image_dir} to {output_dir}")