import os
import json
import xml.etree.ElementTree as ET
import glob

def voc_to_coco(ann_dir, output_file):
    """
    Converts annotations from Pascal VOC format to COCO format.

    :param ann_dir: Path to the directory containing the VOC XML annotation files.
    :param output_file: Path to save the output COCO JSON file.
    """
    # Initialize the main COCO structure
    coco_output = {
        "info": {
            "description": "Converted VOC to COCO format",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}  # To map category names to integer IDs
    category_id_counter = 1
    annotation_id_counter = 1
    image_id_counter = 1

    # Find all XML files in the annotation directory
    xml_files = glob.glob(os.path.join(ann_dir, '*.xml'))
    
    print(f"Found {len(xml_files)} XML files in {ann_dir}")

    # Process each XML file
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # --- Image Information ---
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        image_info = {
            "id": image_id_counter,
            "file_name": filename,
            "width": width,
            "height": height
        }
        coco_output["images"].append(image_info)

        # --- Annotation Information ---
        for obj in root.findall('object'):
            category_name = obj.find('name').text

            # Add new category if it's not already in the map
            if category_name not in category_map:
                category_map[category_name] = category_id_counter
                coco_output["categories"].append({
                    "id": category_id_counter,
                    "name": category_name,
                    "supercategory": "none"
                })
                category_id_counter += 1

            category_id = category_map[category_name]

            # Bounding box conversion
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # COCO format is [xmin, ymin, width, height]
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            bbox = [xmin, ymin, bbox_width, bbox_height]
            
            area = bbox_width * bbox_height

            annotation_info = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0, # Assuming no crowd annotations
                "segmentation": [] # Assuming no segmentation
            }
            coco_output["annotations"].append(annotation_info)
            annotation_id_counter += 1
        
        image_id_counter += 1
    
    # Save the final COCO JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)
    
    print(f"Conversion successful! COCO JSON file saved to {output_file}")


if __name__ == '__main__':
    # --- IMPORTANT: UPDATE THESE PATHS ---
    # NOTE: It's good practice to use raw strings (r"...") or forward slashes for Windows paths
    
    # Path to the directory containing your XML annotation files
    annotation_directory = r"D:\kazemloo\1-domain_adaptation\docker_code\app2\data\tanks\images\test\labels"
    
    # Path where you want to save the output COCO JSON file
    output_json_file = r"D:\kazemloo\1-domain_adaptation\docker_code\app2\data\tanks\images\test\coco_annotations.json"

    # Run the conversion
    voc_to_coco(annotation_directory, output_json_file)