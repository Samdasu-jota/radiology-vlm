import json
import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor



class RadiologyDataset(Dataset):
    def __init__(self, json_path, processor):
        self.processor = processor
        
        # Load all image-text pairs from JSON
        with open(json_path, "r") as f:
            self.pairs = json.load(f)
        
        print(f"Dataset loaded: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # load image from disk
        image = Image.open(pair["image_path"]).convert("RGB")
        text = pair["text"]

        # convert image and text to tensors using CLIP processor
        inputs = self.processor(
            text = text,
            images = image,
            return_tensors = "pt",
            padding = "max_length",
            max_length = 77,
            truncation = True
        )

        # Remove the extra batch dimension the processor adds
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }


if __name__ == "__main__":
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = RadiologyDataset("data/dataset.json", processor)
    
    # Test fetching one sample
    sample = dataset[0]
    print("pixel_values shape:", sample["pixel_values"].shape)
    print("input_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)