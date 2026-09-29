"""Check that the FLIR ADAS v2 training split loads, and print its first sample."""
import argparse
from pathlib import Path

from dataset import FLIRThermalDataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check that the FLIR dataset loads correctly.")
    parser.add_argument("--data-dir", default="extracted_flir/FLIR_ADAS_v2",
                        help="folder that contains images_thermal_train/")
    args = parser.parse_args(argv)

    train_dir = Path(args.data_dir) / "images_thermal_train"
    dataset = FLIRThermalDataset(img_folder=str(train_dir), ann_file=str(train_dir / "coco.json"))
    print(f"Loaded {len(dataset)} thermal images.")

    img, target = dataset[0]
    annotations = target["annotations"]
    print(f"First image: id {target['image_id']}, size {img.size}, mode {img.mode}")
    print(f"Annotated objects: {len(annotations)}")
    print(f"Category IDs: {[a['category_id'] for a in annotations]}")
    if annotations:
        print(f"First box (x, y, width, height): {annotations[0]['bbox']}")


if __name__ == "__main__":
    main()
