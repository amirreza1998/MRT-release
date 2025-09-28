import os
from PIL import Image
from tqdm import tqdm

# --- CONFIGURE THIS ---
# Point this to your image directory (e.g., the 'val' folder)
IMAGE_DIRECTORY = "/app/app4/data/tanks/images/test/images" 
# --------------------

print(f"Scanning directory: {IMAGE_DIRECTORY}")
corrupted_files = []

# Use os.walk to find all files, including in subdirectories if they exist
for root, _, files in os.walk(IMAGE_DIRECTORY):
    for file_name in tqdm(files, desc="Checking images"):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue # Skip non-image files

        file_path = os.path.join(root, file_name)
        try:
            with Image.open(file_path) as img:
                img.load() # This forces a full read of the image data
        except Exception as e:
            print(f"\n--- Found Corrupted File ---")
            print(f"File: {file_path}")
            print(f"Error: {e}")
            print(f"--------------------------\n")
            corrupted_files.append(file_path)

if not corrupted_files:
    print("\n✅ Scan complete. No corrupted files found.")
else:
    print(f"\n❌ Scan complete. Found {len(corrupted_files)} corrupted files:")
    for f in corrupted_files:
        print(f)