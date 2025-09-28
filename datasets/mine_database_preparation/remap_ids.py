import json
import os

# --- 1. CONFIGURATION ---
# UPDATE this path to your JSON file
# json_path = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\annotations\test_cocostyle_val.json'
json_path = r'E:\deep_da_project\1-main_project\data\tanks\annotations\test_cocostyle.json'

# --- 2. SCRIPT LOGIC ---

def remap_coco_ids(original_file_path):
    """
    Remaps category IDs in a COCO JSON file from 1-based to 0-based.
    Specifically maps {1: 0, 2: 1}.
    """
    print(f"Starting remapping process for: {os.path.basename(original_file_path)} ⚙️")

    # Define the mapping from old IDs to new IDs
    # You can extend this if you have more classes, e.g., {1:0, 2:1, 3:2, ...}
    id_map = {
        1: 0,
        2: 1
    }

    # Load the original COCO data
    try:
        with open(original_file_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{original_file_path}'")
        return

    # --- Remap 'categories' section ---
    print("Remapping category IDs in the 'categories' list...")
    for category in coco_data['categories']:
        old_id = category['id']
        if old_id in id_map:
            category['id'] = id_map[old_id]

    # --- Remap 'annotations' section ---
    print("Remapping category IDs in the 'annotations' list...")
    annotations_updated = 0
    for annotation in coco_data['annotations']:
        old_id = annotation['category_id']
        if old_id in id_map:
            annotation['category_id'] = id_map[old_id]
            annotations_updated += 1
    
    print(f"Updated {annotations_updated} annotations.")

    # --- Save the corrected file ---
    output_path = original_file_path.replace('.json', '_remapped.json')
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"\nSuccessfully saved remapped annotations to:\n{output_path} ✅")


if __name__ == '__main__':
    remap_coco_ids(json_path)