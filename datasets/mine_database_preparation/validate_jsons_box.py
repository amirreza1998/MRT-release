import json
import os

# --- CONFIGURE THIS ---
ANNOTATION_FILE = r'E:\deep_da_project\1-main_project\data\tanks\annotations\test_cocostyle.json' 
# --------------------

def validate_coco_annotations(file_path):
    """
    Validates a COCO JSON file for common errors like invalid bboxes and category IDs.
    """
    if not os.path.exists(file_path):
        print(f"Error: Annotation file not found at '{file_path}'")
        return

    print(f"Validating {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    if not all([images, annotations, categories]):
        print("Error: JSON file is missing 'images', 'annotations', or 'categories' key.")
        return

    # Create lookups for faster access
    image_dims = {img['id']: (img['width'], img['height']) for img in images}
    valid_category_ids = {cat['id'] for cat in categories}
    
    print(f"Found {len(images)} images, {len(annotations)} annotations, and {len(categories)} categories.")
    print(f"Valid Category IDs are: {sorted(list(valid_category_ids))}")

    error_count = 0
    for i, ann in enumerate(annotations):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']
        
        # 1. Check if the annotation's image_id exists in the images list
        if image_id not in image_dims:
            print(f"  [ERROR] Annotation ID {ann.get('id', i)} has an image_id ({image_id}) that does not exist in the 'images' list.")
            error_count += 1
            continue # Skip other checks for this annotation

        img_w, img_h = image_dims[image_id]
        x, y, w, h = bbox

        # 2. Check for non-positive width or height
        if w <= 0 or h <= 0:
            print(f"  [ERROR] Annotation ID {ann.get('id', i)} (Image ID: {image_id}) has a non-positive width/height.")
            print(f"          -> BBox: [{x}, {y}, {w}, {h}]")
            error_count += 1

        # 3. Check if the bounding box is within the image boundaries
        if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
            print(f"  [WARNING] Annotation ID {ann.get('id', i)} (Image ID: {image_id}) is partially or fully outside image boundaries.")
            print(f"            -> BBox: [{x}, {y}, {w}, {h}] vs Image Size: [{img_w}, {img_h}]")
            # This is often a warning, but can cause issues with some augmentation libraries.

        # 4. Check for invalid category IDs
        if category_id not in valid_category_ids:
            print(f"  [ERROR] Annotation ID {ann.get('id', i)} (Image ID: {image_id}) has an invalid Category ID.")
            print(f"          -> Category ID: {category_id} is not in the list of valid IDs.")
            error_count += 1
    
    print("-" * 30)
    if error_count == 0:
        print("Validation complete. No critical errors found! ✅")
    else:
        print(f"Validation complete. Found {error_count} critical error(s). Please fix them in your JSON file. ❌")


if __name__ == '__main__':
    validate_coco_annotations(ANNOTATION_FILE)