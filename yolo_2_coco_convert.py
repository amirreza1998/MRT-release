import json
import os
import glob
from PIL import Image
from pathlib import Path


class YOLOtoCOCOConverter:
    def __init__(self, dataset_root, output_root, class_names=None):
        """
        Convert YOLO format dataset to COCO format for MRT-release model
        Handles structure: train/image/ and train/label/
        
        Args:
            dataset_root (str): Root directory containing train/ folder
            output_root (str): Output directory for COCO format dataset
            class_names (list): List of class names. If None, will use generic names
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.class_names = class_names or ["class_0"]  # Default class name
        
        # Create output directory structure
        self.setup_output_structure()
        
    def setup_output_structure(self):
        """Create the required directory structure"""
        (self.output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (self.output_root / "images" / "target").mkdir(parents=True, exist_ok=True)
        (self.output_root / "annotations").mkdir(parents=True, exist_ok=True)
        
    def yolo_to_coco_bbox(self, yolo_bbox, img_width, img_height):
        """
        Convert YOLO bbox format to COCO bbox format
        YOLO: [x_center, y_center, width, height] (normalized)
        COCO: [x_min, y_min, width, height] (absolute pixels)
        """
        x_center, y_center, width, height = yolo_bbox
        
        # Convert from normalized to absolute coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Convert to COCO format (top-left corner)
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        
        return [x_min, y_min, width, height]
    
    def read_yolo_annotation(self, txt_file):
        """Read YOLO format annotation file"""
        annotations = []
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        annotations.append((class_id, bbox))
        return annotations
    
    def create_coco_annotation(self, image_id, annotation_id, class_id, bbox, img_width, img_height):
        """Create a single COCO annotation"""
        coco_bbox = self.yolo_to_coco_bbox(bbox, img_width, img_height)
        area = coco_bbox[2] * coco_bbox[3]  # width * height
        
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id + 1,  # COCO categories start from 1
            "bbox": coco_bbox,
            "area": area,
            "iscrowd": 0
        }
    
    def process_train_split(self):
        """Process the train split with image/ and label/ subfolders"""
        train_dir = self.dataset_root / "train"
        image_dir = train_dir / "images"
        label_dir = train_dir / "labels"
        output_img_dir = self.output_root / "images" / "train"
        
        if not image_dir.exists():
            print(f"Error: {image_dir} does not exist!")
            return self.create_empty_coco_json()
            
        if not label_dir.exists():
            print(f"Warning: {label_dir} does not exist! Creating annotations without labels.")
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(image_dir / ext)))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return self.create_empty_coco_json()
        
        # Initialize COCO format structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i+1, "name": name} for i, name in enumerate(self.class_names)]
        }
        
        annotation_id = 1
        
        for image_id, image_path in enumerate(image_files, 1):
            image_path = Path(image_path)
            
            # Copy image to output directory
            output_image_path = output_img_dir / image_path.name
            if not output_image_path.exists():
                import shutil
                shutil.copy2(image_path, output_image_path)
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_path.name,
                "width": img_width,
                "height": img_height
            })
            
            # Find corresponding label file
            # Convert image filename to label filename (same name, .txt extension)
            label_filename = image_path.stem + '.txt'  # Remove extension and add .txt
            label_path = label_dir / label_filename
            
            # Read YOLO annotations
            yolo_annotations = self.read_yolo_annotation(label_path)
            
            # Convert annotations
            for class_id, bbox in yolo_annotations:
                coco_annotation = self.create_coco_annotation(
                    image_id, annotation_id, class_id, bbox, img_width, img_height
                )
                coco_data["annotations"].append(coco_annotation)
                annotation_id += 1
        
        return coco_data
    
    def create_empty_coco_json(self):
        """Create empty COCO JSON structure"""
        return {
            "images": [],
            "annotations": [],
            "categories": [{"id": i+1, "name": name} for i, name in enumerate(self.class_names)]
        }
    
    def create_empty_splits(self):
        """Create empty test and target splits"""
        empty_data = self.create_empty_coco_json()
        
        # Create empty test annotation
        test_file = self.output_root / "annotations" / "test_cocostyle.json"
        with open(test_file, 'w') as f:
            json.dump(empty_data, f, indent=2)
        
        # Create empty target annotation
        target_file = self.output_root / "annotations" / "target_cocostyle.json"
        with open(target_file, 'w') as f:
            json.dump(empty_data, f, indent=2)
            
        print(f"Created empty test and target annotation files")
    
    def convert(self):
        """Convert the train dataset only"""
        print("Processing train split...")
        coco_data = self.process_train_split()
        
        # Save train COCO annotation file
        output_file = self.output_root / "annotations" / "train_cocostyle.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Saved {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {output_file}")
        
        # Create empty test and target files
        self.create_empty_splits()
        
        print(f"\nConversion complete! Dataset saved to: {self.output_root}")
        self.print_usage_info()
    
    def print_usage_info(self):
        """Print information about how to use the converted dataset"""
        print("\n" + "="*60)
        print("USAGE INFORMATION")
        print("="*60)
        print(f"Your dataset has been converted to: {self.output_root}")
        print("\nTo use with MRT-release model, update the dataset configuration:")
        print("""
In the modified coco_style_dataset.py, use:

img_dirs = {
    'your_dataset': {
        'train': 'your_dataset/images/train',
        'test': 'your_dataset/images/test', 
        'target': 'your_dataset/images/target'
    }
}

anno_files = {
    'your_dataset': {
        'source': {
            'train': 'your_dataset/annotations/train_cocostyle.json',
            'test': 'your_dataset/annotations/test_cocostyle.json',
        },
        'target': {
            'target': 'your_dataset/annotations/target_cocostyle.json'
        }
    }
}

Usage in training:
dataset = CocoStyleDataset(
    root_dir="/path/to/your/converted/dataset",
    dataset_name="your_dataset",
    domain="source",
    split="train",
    transforms=your_transforms
)
        """)


def main():
    """Main function with example usage"""
    # Example usage
    dataset_root = r"D:\kazemloo\1-domain_adaptation\docker_volume\data"  # Contains train/ folder with image/ and label/ subfolders
    output_root = r"D:\kazemloo\1-domain_adaptation\docker_volume\train_coco"  # Where to save COCO format dataset
    class_names = ["your_class_name"]  # Replace with your actual class names
    
    # If you have multiple classes, list them all:
    # class_names = ["person", "car", "bicycle", ...]
    
    converter = YOLOtoCOCOConverter(dataset_root, output_root, class_names)
    converter.convert()


if __name__ == "__main__":

    main()