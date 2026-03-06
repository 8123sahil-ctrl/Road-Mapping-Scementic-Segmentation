import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# -----------------------------
# Dataset: image.jpg + image_mask.png in same folder
# -----------------------------
class RoboflowSegDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.images = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.endswith("_mask.png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)

        # mask name format: <image_name>_mask.png
        base, _ = os.path.splitext(img_name)
        mask_name = base + "_mask.png"
        mask_path = os.path.join(self.folder, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        # -----------------------------
        # SIZE LOGIC (added)
        # -----------------------------
        TARGET_H, TARGET_W = 512, 512
        image = cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW

        # Roboflow masks are usually 0 background, 255 road. Convert to 0/1.
        mask = (mask > 0).astype(np.int64)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

def main():
    train_folder = "data/train"
    val_folder   = "data/valid"

    train_ds = RoboflowSegDataset(train_folder)
    val_ds   = RoboflowSegDataset(val_folder)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    # DeepLabV3+
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,   # background + road
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20
    best_val = 1e9

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            out = model(imgs)  # [B,2,H,W]
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                loss = criterion(out, masks)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "deeplabv3plus_road_best.pth")
            print("Saved: deeplabv3plus_road_best.pth")

    print("Training done.")

if __name__ == "__main__":
    main()

