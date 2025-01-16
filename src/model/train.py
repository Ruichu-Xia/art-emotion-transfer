import os 

import torch
from tqdm import tqdm


def train_model_single_epoch(
    model,
    train_loader,
    custom_loss,
    optimizer,
    device,
    scaler,
    grad_clip,
    triplet_weight=0.5,  # Weight for triplet loss
    margin=1.0,  # Margin for triplet loss
):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=str(device), dtype=torch.float16):
            embeddings, logits = model(images)
            loss = custom_loss(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=grad_clip)
            
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        progress_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss


def validate_model_single_epoch(
    model,
    val_loader,
    custom_loss,
    device,
    triplet_weight=0.5,  # Weight for triplet loss
    margin=1.0,  # Margin for triplet loss
):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            embeddings, logits = model(images)
            loss = custom_loss(logits, labels)

            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss


def save_checkpoint(epoch, model, optimizer, history, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{epoch}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")