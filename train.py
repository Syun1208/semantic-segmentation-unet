import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from tabulate import tabulate

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 80  # 1280 originally
IMAGE_WIDTH = 160  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "/content/data_v2/data_v2/train"
TRAIN_MASK_DIR = "/content/data_v2/data_v2/trainanot"
VAL_IMG_DIR = "/content/data_v2/data_v2/val"
VAL_MASK_DIR = "/content/data_v2/data_v2/valanot"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("/content/gdrive/MyDrive/Colab Notebooks/my_checkpoint.pth"), model, optimizer)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        clear_output(wait=True)
        # print(f"Epoch {epoch+1}/{NUM_EPOCHS}: ")
        num_correct, num_pixels, dice_score = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        dice_score_result = dice_score/len(train_loader)
        accuracy = num_correct/num_pixels*100
        tables = ['Epoch', 'Accuracy', 'Dice score']
        headers = [['{epoch+1}/{NUM_EPOCHS}', '{accuracy:.2f}', '{dice_score_result}']]
        print(tabulate(headers, tables, tablefmt="psql"))
        # print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
        # print(f"Dice score: {dice_score_result}")
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="/content/saved_images", device=DEVICE
        )
        #plot results
        plt.figure(figsize=(10,8))
        plt.title("Results for UNET segmentation")
        plt.plot(epoch, dice_score_result, '-g')
        plt.plot(epoch, accuracy, 'b')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy and Dice Score")
        plt.legend(['Dice score', 'Accuracy'])
        plt.show()

if __name__ == "__main__":
    main()
