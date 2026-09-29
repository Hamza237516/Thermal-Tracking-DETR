"""Fine-tune DETR-ResNet50 on the Teledyne FLIR ADAS v2 thermal dataset.

Full run (15 epochs, saves a checkpoint after every epoch):
    python train.py

Quick local check (1 epoch, stops after 20 batches, saves nothing):
    python train.py --smoke-test
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

from dataset import FLIRThermalDataset, collate_fn

MODEL_NAME = "facebook/detr-resnet-50"
SMOKE_TEST_BATCHES = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune DETR on FLIR ADAS v2 thermal images.")
    parser.add_argument("--data-dir", default="extracted_flir/FLIR_ADAS_v2",
                        help="folder that contains images_thermal_train/")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="checkpoints",
                        help="where thermal_detr_epoch_<N>.pth files are saved")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="auto uses CUDA when available, otherwise CPU")
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"run 1 epoch, stop after {SMOKE_TEST_BATCHES} batches and save nothing")
    return parser.parse_args(argv)


def pick_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main(argv=None):
    args = parse_args(argv)
    epochs = 1 if args.smoke_test else args.epochs
    device = pick_device(args.device)
    print(f"Device: {device} | epochs: {epochs} | batch size: {args.batch_size} | learning rate: {args.lr}")

    # Model: DETR with a ResNet-50 backbone, pretrained on COCO.
    # FLIR ADAS v2 labels objects with COCO-style category IDs, so we keep DETR's
    # 91-label COCO classification head; a smaller head has no slot for the larger IDs.
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    model = DetrForObjectDetection.from_pretrained(MODEL_NAME, num_labels=91, ignore_mismatched_sizes=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Data: the FLIR thermal training split and its COCO-format annotations.
    train_dir = Path(args.data_dir) / "images_thermal_train"
    dataset = FLIRThermalDataset(img_folder=str(train_dir), ann_file=str(train_dir / "coco.json"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"Training images: {len(dataset)}")

    output_dir = Path(args.output_dir)
    if not args.smoke_test:
        output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss, steps = 0.0, 0
        for batch_idx, (images, targets) in enumerate(loader):
            inputs = processor(images=list(images), annotations=list(targets), return_tensors="pt")
            labels = [{key: value.to(device) for key, value in target.items()} for target in inputs["labels"]]
            pixel_mask = inputs.get("pixel_mask")
            outputs = model(
                pixel_values=inputs["pixel_values"].to(device),
                pixel_mask=pixel_mask.to(device) if pixel_mask is not None else None,
                labels=labels,
            )
            # DETR's Hungarian-matching loss: class, box L1 and GIoU terms, weighted.
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{epochs} | batch {batch_idx}/{len(loader)} | loss {loss.item():.4f}")
            if args.smoke_test and batch_idx + 1 >= SMOKE_TEST_BATCHES:
                break

        print(f"Epoch {epoch} finished | mean loss {running_loss / max(steps, 1):.4f}")
        if args.smoke_test:
            print("Smoke test passed: data loading, forward and backward passes all work.")
        else:
            checkpoint = output_dir / f"thermal_detr_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint)
            print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
