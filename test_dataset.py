import torch
from dataset import FLIRThermalDataset

# We point directly to the unzipped FLIR thermal training folder
img_dir = "extracted_flir/FLIR_ADAS_v2/images_thermal_train"
ann_file = "extracted_flir/FLIR_ADAS_v2/images_thermal_train/coco.json"

print("Initializing PyTorch Dataset...")

try:
    # Load the dataset
    dataset = FLIRThermalDataset(img_folder=img_dir, ann_file=ann_file)
    
    print(f"SUCCESS: Loaded {len(dataset)} thermal images!")
    
    # Grab the very first image and its targets (bounding boxes & labels)
    img, target = dataset[0]
    
    print("\n--- First Image Data ---")
    print(f"Image Size (Pixels): {img.size}")
    print(f"Number of objects detected in this image: {len(target['boxes'])}")
    print(f"Class IDs: {target['labels']}")
    print(f"Bounding Box Coordinates:\n{target['boxes']}")
    
except Exception as e:
    print(f"\nERROR: Something went wrong loading the data.\n{e}")