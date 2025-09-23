import cv2
import json
import os
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# Update these paths to match your dataset
json_path = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\annotations\train_cocostyle_val.json'
image_folder = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\images\train'
output_folder = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\images\test\labelized5'
# -------------------------

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print("Loading JSON annotations...")
with open(json_path, 'r') as f:
    data = json.load(f)

# Create a dictionary to map category IDs to category names
category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

# Group annotations by image ID for efficient processing
print("Grouping annotations by image...")
annotations_by_image_id = {}
for anno in tqdm(data['annotations'], desc="Processing annotations"):
    image_id = anno['image_id']
    if image_id not in annotations_by_image_id:
        annotations_by_image_id[image_id] = []
    annotations_by_image_id[image_id].append(anno)

# --- 2. VISUALIZATION ---
print("\nDrawing bounding boxes on images...")
for image_info in tqdm(data['images'], desc="Visualizing images"):
    image_id = image_info['id']
    file_name = image_info['file_name']
    image_path = os.path.join(image_folder, file_name)

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at {image_path}. Skipping.")
        continue

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        continue

    # Draw all annotations for this image
    if image_id in annotations_by_image_id:
        for anno in annotations_by_image_id[image_id]:
            # Get bounding box coordinates
            x, y, width, height = [int(val) for val in anno['bbox']]
            
            # Get category name
            category_id = anno['category_id']
            category_name = category_id_to_name.get(category_id, 'Unknown')

            # Define colors and draw the box and label
            color = (0, 255, 0) # Green
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save the visualized image
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, image)

print(f"\n✅ Visualization complete. Labeled images are saved in: {output_folder}")