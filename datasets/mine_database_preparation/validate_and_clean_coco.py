import json
import os

def clean_coco_annotations(input_file, output_file):
    """
    Reads a COCO annotation file, removes annotations with bounding boxes
    that are outside their corresponding image's boundaries, and saves
    the cleaned data to a new file.

    Args:
        input_file (str): Path to the input COCO JSON file.
        output_file (str): Path to save the cleaned COCO JSON file.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    print(f"Loading annotations from: {input_file}")
    with open(input_file, 'r') as f:
        coco_data = json.load(f)

    # Extract the main components
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # Create a dictionary for quick lookup of image dimensions (width, height)
    image_dims = {img['id']: (img['width'], img['height']) for img in images}

    cleaned_annotations = []
    removed_count = 0

    print("Scanning annotations to remove invalid entries...")
    # Loop through all annotations and keep only the valid ones
    for ann in annotations:
        image_id = ann['image_id']
        bbox = ann['bbox']
        
        # Check if the image ID for this annotation exists
        if image_id not in image_dims:
            print(f"  [REMOVING] Annotation ID {ann.get('id')} because its image_id ({image_id}) was not found.")
            removed_count += 1
            continue

        img_w, img_h = image_dims[image_id]
        x, y, w, h = bbox

        # Condition to check if the box is valid and within boundaries
        # 1. Width and height must be positive.
        # 2. The top-left corner (x, y) must be within the image.
        # 3. The bottom-right corner (x+w, y+h) must be within the image.
        is_valid = (w > 0 and h > 0 and 
                    x >= 0 and y >= 0 and 
                    (x + w) <= img_w and 
                    (y + h) <= img_h)

        if is_valid:
            cleaned_annotations.append(ann)
        else:
            print(f"  [REMOVING] Annotation ID {ann.get('id')} on Image ID {image_id} is outside boundaries.")
            print(f"             -> BBox: [{x}, {y}, {w}, {h}] vs Image Size: [{img_w}, {img_h}]")
            removed_count += 1

    # Create the new dataset with the cleaned annotations
    cleaned_coco_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': coco_data['images'],
        'annotations': cleaned_annotations, # Use the filtered list
        'categories': coco_data['categories']
    }

    print("-" * 30)
    print(f"Removed {removed_count} invalid annotation(s).")
    print(f"Kept {len(cleaned_annotations)} valid annotation(s).")
    
    # Save the cleaned data to the new file
    print(f"Saving cleaned annotations to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(cleaned_coco_data, f, indent=4)
        
    print("Cleaning complete! ✅")


# --- Main execution ---
if __name__ == '__main__':
    # Configure your file paths here
    INPUT_JSON_FILE = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\annotations\train_cocostyle_val_dirty.json' #train_cocostyle_val_dirty.json
    CLEANED_OUTPUT_JSON_FILE = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\annotations\train_cocostyle_val.json' #train_cocostyle_val.json
    
    clean_coco_annotations(
        input_file=INPUT_JSON_FILE,
        output_file=CLEANED_OUTPUT_JSON_FILE
    )