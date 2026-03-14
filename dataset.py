import torchvision

class FLIRThermalDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file):
        # PyTorch natively reads the FLIR coco.json file
        super(FLIRThermalDataset, self).__init__(img_folder, ann_file)

    def __getitem__(self, idx):
        # 1. Grab the raw image and raw COCO target dictionary
        img, target = super(FLIRThermalDataset, self).__getitem__(idx)
        
        # 2. Package it exactly how the Hugging Face Processor demands it:
        # {"image_id": ID, "annotations": [raw_coco_objects]}
        image_id = self.ids[idx]
        formatted_target = {'image_id': image_id, 'annotations': target}
        
        return img, formatted_target

# --- The Collate Function ---
def collate_fn(batch):
    # Safely bundles our images and targets into batches of 2
    return tuple(zip(*batch))