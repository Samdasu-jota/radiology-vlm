import torch
import json
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from src.dataset import RadiologyDataset


CONFIG = {
    "json_path": "data/dataset.json",
    "model_name": "openai/clip-vit-base-patch32",
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 1e-5,
    "num_workers": 2,
    "save_path": "model/fine_tuned_clip"
}

def contrastive_loss(logits):
    labels = torch.arange(len(logits), device= logits.device)
    loss_images = torch.nn.functional.cross_entropy(logits, labels)
    loss_texts = torch.nn.functional.cross_entropy(logits.T, labels)
    return (loss_images + loss_texts) / 2


def train():
    # Set up device - use MPS (Apple GPU) if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU) ✅")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load model and processor
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(CONFIG["model_name"])
    processor = CLIPProcessor.from_pretrained(CONFIG["model_name"])
    model = model.to(device)

    # Load dataset and dataloader
    dataset = RadiologyDataset(CONFIG["json_path"], processor)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )

    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    # Training loop
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    print("-" * 50)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Calculate loss
            logits = outputs.logits_per_image
            loss = contrastive_loss(logits)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}  |  Loss: {avg_loss:.4f}")

    # Save the model
    print("-" * 50)
    print("Saving model...")
    import os
    os.makedirs(CONFIG["save_path"], exist_ok=True)
    model.save_pretrained(CONFIG["save_path"])
    processor.save_pretrained(CONFIG["save_path"])
    print(f"Model saved to {CONFIG['save_path']} ✅")


if __name__ == "__main__":
    train()





