import os
import json

# --- 1. CONFIGURATION ---
# IMPORTANT: Update these two paths to match your project.
# These should be the same paths you used in your original conversion script.

# Path to the directory containing your XML annotation files
annotation_directory = r"C:\Users\KianTejaratCo\Desktop\16 sep\data\tanks\images\test\labels"

# Path to the generated COCO JSON file that you want to fix
coco_json_file = r"C:\Users\KianTejaratCo\Desktop\16 sep\data\tanks\images\test\coco_annotations.json"


# --- 2. SCRIPT LOGIC (No need to edit below this line) ---

def fix_coco_filenames(ann_dir, json_file):
    """
    Corrects the 'file_name' entries in a COCO JSON file to match the
    filenames of the source XML annotation files.
    """
    print("Starting filename correction process... ⚙️")

    # --- Load JSON Data ---
    try:
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        print(f"Successfully loaded '{os.path.basename(json_file)}'.")
    except FileNotFoundError:
        print(f"ERROR: The JSON file was not found at '{json_file}'")
        return

    # --- Get Correct Filenames from Annotation Directory ---
    try:
        # Get a sorted list of all .xml files
        xml_files = sorted([f for f in os.listdir(ann_dir) if f.lower().endswith('.xml')])
        if not xml_files:
            print(f"ERROR: No .xml files found in '{ann_dir}'. Please check the path.")
            return
        print(f"Found {len(xml_files)} XML files in the annotation directory.")
    except FileNotFoundError:
        print(f"ERROR: The annotation directory was not found at '{ann_dir}'")
        return

    # Sort the images list in the COCO data by 'id' to ensure consistent order
    coco_images = sorted(coco_data['images'], key=lambda img: img['id'])

    # --- Verification Step ---
    if len(coco_images) != len(xml_files):
        print("\n--- WARNING! ---")
        print("Mismatch in file count!")
        print(f"The JSON file has {len(coco_images)} image entries.")
        print(f"The directory has {len(xml_files)} XML files.")
        print("Please ensure the JSON file was generated from this exact annotation directory.")
        return

    # --- Update Logic ---
    updated_count = 0
    for i, image_info in enumerate(coco_images):
        old_name = image_info['file_name']
        
        # Create the correct filename by replacing the .xml extension with .jpg
        correct_name = os.path.splitext(xml_files[i])[0] + '.jpg'

        if old_name != correct_name:
            image_info['file_name'] = correct_name
            updated_count += 1
            # Optional: Uncomment the line below to see every change printed to the console
            # print(f"  Updating '{old_name}' -> '{correct_name}'")

    # The list was sorted, but the update happens on the original objects in coco_data
    
    # --- Save the Corrected File ---
    if updated_count > 0:
        output_path = json_file.replace('.json', '_corrected.json')
        print(f"\nSuccessfully updated {updated_count} filenames.")
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        print(f"Corrected annotations saved to: {output_path} ✅")
    else:
        print("\nNo filenames needed to be updated. Everything seems correct.")


if __name__ == '__main__':
    fix_coco_filenames(annotation_directory, coco_json_file)