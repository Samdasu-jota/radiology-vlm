import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from src.dataset_negation import NegationAwareDataset


CONFIG = {
    "json_path": "data/dataset_structured.json",
    "base_model": "model/fine_tuned_clip",
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 5e-6,
    "num_workers": 2,
    "save_path": "model/negation_aware_clip",
    "margin": 0.2,
    "neg_weight": 0.5,
}


def get_embeddings(model, pixel_values, input_ids, attention_mask):
    """Get normalized image and text embeddings using model internals."""
    # Image embeddings
    vision_out = model.vision_model(pixel_values=pixel_values)
    image_embeds = model.visual_projection(vision_out.pooler_output)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    # Text embeddings
    text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    text_embeds = model.text_projection(text_out.pooler_output)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    return image_embeds, text_embeds


def negation_aware_loss(model, pixel_values, input_ids, attention_mask,
                        neg_input_ids, neg_attention_mask, margin, neg_weight):
    """Combined contrastive + hard negative margin loss."""
    image_embeds, pos_text_embeds = get_embeddings(
        model, pixel_values, input_ids, attention_mask
    )

    # Term 1: Standard symmetric contrastive loss
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_embeds @ pos_text_embeds.T
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_image.T, labels)
    contrastive_loss = (loss_i2t + loss_t2i) / 2

    # Term 2: Hard negative margin loss
    # Get embeddings for the flipped (negated) text
    text_out = model.text_model(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
    neg_text_embeds = model.text_projection(text_out.pooler_output)
    neg_text_embeds = neg_text_embeds / neg_text_embeds.norm(dim=-1, keepdim=True)

    # Per-sample similarities
    pos_sim = (image_embeds * pos_text_embeds).sum(dim=-1)  # (B,)
    neg_sim = (image_embeds * neg_text_embeds).sum(dim=-1)  # (B,)

    # Margin loss: penalize when negative is too close to (or closer than) positive
    hard_neg_loss = torch.relu(neg_sim - pos_sim + margin).mean()

    total_loss = contrastive_loss + neg_weight * hard_neg_loss
    return total_loss, contrastive_loss.item(), hard_neg_loss.item()


def train():
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load the already fine-tuned model as starting point
    print(f"Loading base model from {CONFIG['base_model']}...")
    model = CLIPModel.from_pretrained(CONFIG["base_model"])
    processor = CLIPProcessor.from_pretrained(CONFIG["base_model"])
    model = model.to(device)

    # Dataset and dataloader
    dataset = NegationAwareDataset(CONFIG["json_path"], processor)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    print(f"\nTraining for {CONFIG['epochs']} epochs")
    print(f"  margin={CONFIG['margin']}, neg_weight={CONFIG['neg_weight']}")
    print("-" * 60)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        total_contrastive = 0
        total_hard_neg = 0
        num_batches = 0

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)

            loss, c_loss, h_loss = negation_aware_loss(
                model, pixel_values, input_ids, attention_mask,
                neg_input_ids, neg_attention_mask,
                CONFIG["margin"], CONFIG["neg_weight"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contrastive += c_loss
            total_hard_neg += h_loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_c = total_contrastive / num_batches
        avg_h = total_hard_neg / num_batches
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}  |  "
              f"Loss: {avg_loss:.4f}  (contrastive: {avg_c:.4f}, hard_neg: {avg_h:.4f})")

    # Save
    print("-" * 60)
    os.makedirs(CONFIG["save_path"], exist_ok=True)
    model.save_pretrained(CONFIG["save_path"])
    processor.save_pretrained(CONFIG["save_path"])
    print(f"Model saved to {CONFIG['save_path']}")


if __name__ == "__main__":
    train()
