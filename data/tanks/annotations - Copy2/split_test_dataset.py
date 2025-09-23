import json
import random
import os

def split_coco_dataset(
    input_json_path,
    train_output_path,
    val_output_path,
    split_ratio=0.8,
    random_seed=42
):
    """
    Splits a COCO-style JSON dataset into training and validation sets.

    Args:
        input_json_path (str): Path to the input COCO JSON file.
        train_output_path (str): Path to save the output training JSON file.
        val_output_path (str): Path to save the output validation JSON file.
        split_ratio (float): The ratio of the dataset to be used for training (e.g., 0.8 for 80%).
        random_seed (int): Seed for the random number generator for reproducibility.
    """
    print(f"Loading dataset from: {input_json_path}")
    
    # 1. Load the original COCO JSON file
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract the main components from the COCO data
    info = coco_data.get('info', {})
    licenses = coco_data.get('licenses', [])
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])

    # 2. Shuffle the images randomly for an unbiased split
    if random_seed:
        random.seed(random_seed)
    random.shuffle(images)

    # 3. Calculate the split index
    split_index = int(len(images) * split_ratio)

    # 4. Split the images into training and validation sets
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create sets of image IDs for quick lookup
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    
    print(f"Splitting dataset: {len(train_image_ids)} train images and {len(val_image_ids)} validation images.")

    # 5. Split annotations based on the image IDs
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    
    print(f"Found {len(train_annotations)} annotations for training and {len(val_annotations)} for validation.")

    # 6. Create the new COCO JSON structures
    train_coco = {
        'info': info,
        'licenses': licenses,
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories,
    }

    val_coco = {
        'info': info,
        'licenses': licenses,
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories,
    }

    # 7. Save the new JSON files
    print(f"Saving training set to: {train_output_path}")
    with open(train_output_path, 'w') as f:
        json.dump(train_coco, f, indent=4)

    print(f"Saving validation set to: {val_output_path}")
    with open(val_output_path, 'w') as f:
        json.dump(val_coco, f, indent=4)
        
    print("Splitting complete! ✅")

# --- Main execution ---
if __name__ == '__main__':
    # Define your file paths here
    INPUT_JSON = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\annotations\target_cocostyle.json'
    TRAIN_JSON = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\annotations\target_cocostyle_train.json'
    VAL_JSON = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\annotations\target_cocostyle_val.json'
    
    # Define the split ratio (80% for training, 20% for validation)
    TRAIN_RATIO = 0.8
    
    # Check if the input file exists
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Input file not found at '{INPUT_JSON}'")
    else:
        split_coco_dataset(
            input_json_path=INPUT_JSON,
            train_output_path=TRAIN_JSON,
            val_output_path=VAL_JSON,
            split_ratio=TRAIN_RATIO
        )