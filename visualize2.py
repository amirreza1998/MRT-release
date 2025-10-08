import cv2
import xml.etree.ElementTree as ET
import os

# --- 1. CONFIGURATION ---
# Update these paths to point to your image and its corresponding XML file
image_path = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\images\test\images\pi_Fuchs_Tpz_1_6x6_wheeled_armoured_vehicle_personnel_carrier_Rheinmetall_Germany_German_army_46.jpg'
xml_path = r'D:\kazemloo\1-domain_adaptation\docker_code\app4\data\tanks\images\test\labels\pi_Fuchs_Tpz_1_6x6_wheeled_armoured_vehicle_personnel_carrier_Rheinmetall_Germany_German_army_46.xml'
# -------------------------

# Check if files exist
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit()
if not os.path.exists(xml_path):
    print(f"Error: XML file not found at {xml_path}")
    exit()

# Load the image
image = cv2.imread(image_path)

# Parse the XML file
tree = ET.parse(xml_path)
root = tree.getroot()

# --- 2. DRAWING BOUNDING BOXES ---
# Find all 'object' tags in the XML and draw a box for each
for member in root.findall('object'):
    # Get the class name
    class_name = member.find('name').text

    # Get the bounding box coordinates
    bndbox = member.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    
    # Define color and draw the rectangle
    color = (0, 255, 0) # Green
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    
    # Draw the label text above the rectangle
    cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

print("✅ Visualization complete. Press any key to close the image window.")

# --- 3. DISPLAY THE RESULT ---
# Show the image in a new window
cv2.imshow('Image with Bounding Boxes', image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- (Optional) 4. SAVE THE RESULT ---
# If you want to save the file instead of displaying it,
# comment out section 3 and uncomment the line below.
# cv2.imwrite('./output_image.jpg', image)
# print("Image saved as output_image.jpg")