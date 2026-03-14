import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from dataset import FLIRThermalDataset, collate_fn

print("Initializing DETR Training Pipeline...")

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 2      
EPOCHS = 1          
LEARNING_RATE = 1e-4

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
# THIS IS THE LINE THAT WENT MISSING!
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=91, # Fixed to 91 to prevent the index error
    ignore_mismatched_sizes=True 
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 3. DATA LOADER 
# ==========================================
img_dir = "extracted_flir/FLIR_ADAS_v2/images_thermal_train"
ann_file = "extracted_flir/FLIR_ADAS_v2/images_thermal_train/coco.json"

train_dataset = FLIRThermalDataset(img_folder=img_dir, ann_file=ann_file)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn
)

# ==========================================
# 4. THE TRAINING LOOP
# ==========================================
print(f"\nStarting Training! Total Epochs: {EPOCHS}")

model.train()

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        
        inputs = processor(images=list(images), annotations=list(targets), return_tensors="pt")
        outputs = model(**inputs)
        
        loss_dict = outputs.loss_dict
        total_loss = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_dataloader)} | Loss: {total_loss.item():.4f}")
            
        if batch_idx == 20:
            print("\nLocal test successful! Stopping early.")
            break 
            
print("Training Loop Test Complete!")