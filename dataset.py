import torchvision


class FLIRThermalDataset(torchvision.datasets.CocoDetection):
    """FLIR ADAS v2 thermal images with their COCO-format annotations.

    Each item comes back in the format Hugging Face's DetrImageProcessor expects:
    (PIL image, {"image_id": id, "annotations": [COCO annotation dicts]}).
    torchvision opens the single-channel thermal frames as 3-channel RGB,
    which matches the input the COCO-pretrained DETR backbone was trained on.
    """

    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)

    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        return img, {"image_id": image_id, "annotations": annotations}


def collate_fn(batch):
    """Keep images and targets as tuples; the image processor batches them later."""
    return tuple(zip(*batch))
