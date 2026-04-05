import torch
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor


def load_models():
    print("Loading vanilla CLIP...")
    vanilla_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    vanilla_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading fine-tuned CLIP...")
    finetuned_model = CLIPModel.from_pretrained("model/fine_tuned_clip")
    finetuned_processor = CLIPProcessor.from_pretrained("model/fine_tuned_clip")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU) ✅")
    else:
        device = torch.device("cpu")

    vanilla_model = vanilla_model.to(device)
    finetuned_model = finetuned_model.to(device)

    # Set to evaluation mode
    vanilla_model.eval()
    finetuned_model.eval()

    return vanilla_model, vanilla_processor, finetuned_model, finetuned_processor, device

def build_index(model, processor, dataset_json, device):
    print("Building embedding index...")
    
    with open(dataset_json, "r") as f:
        pairs = json.load(f)
    
    embeddings = []
    image_paths = []
    texts = []
    
    model.eval()
    with torch.no_grad():
        for i, pair in enumerate(pairs):
            try:
                image = Image.open(pair["image_path"]).convert("RGB")
            except:
                continue
            
            # Process image - only get pixel_values
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            
            # Get image embedding using only pixel_values
            vision_output = model.vision_model(pixel_values=pixel_values)
            image_embedding = vision_output.pooler_output
            image_embedding = model.visual_projection(image_embedding)
            
            # Normalize
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_embedding.cpu().numpy())
            image_paths.append(pair["image_path"])
            texts.append(pair["text"])
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(pairs)} images")
    
    embeddings = np.vstack(embeddings)
    print(f"Index built: {len(image_paths)} embeddings of shape {embeddings.shape}")
    
    return embeddings, image_paths, texts

def retrieve(query_text, model, processor, embeddings, image_paths, texts, device, top_k=5):
    print("  Step 1: setting eval mode...")
    model.eval()
    
    print("  Step 2: processing text...")
    with torch.no_grad():
        inputs = processor(
            text=query_text,
            return_tensors="pt",
            padding=True
        )
        
        print("  Step 3: moving to device...")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        print("  Step 4: getting text features...")
        text_output = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        print("  Step 5: projecting...")
        query_embedding = text_output.pooler_output
        query_embedding = model.text_projection(query_embedding)
        
        print("  Step 6: normalizing...")
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy()
    
    print("  Step 7: computing similarities...")
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    
    print("  Step 8: sorting results...")
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "image_path": image_paths[idx],
            "text": texts[idx],
            "score": float(similarities[idx])
        })
    
    print("  Done!")
    return results

def display_results(query_text, vanilla_results, finetuned_results):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Query: "{query_text}"', fontsize=14, fontweight='bold')
    
    # Top row = vanilla CLIP results
    for i, result in enumerate(vanilla_results):
        img = Image.open(result["image_path"]).convert("RGB")
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"Score: {result['score']:.3f}", fontsize=9)
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("Vanilla CLIP", fontsize=12, fontweight='bold')
    
    # Bottom row = fine-tuned results
    for i, result in enumerate(finetuned_results):
        img = Image.open(result["image_path"]).convert("RGB")
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(f"Score: {result['score']:.3f}", fontsize=9)
        axes[1, i].axis("off")
    axes[1, 0].set_ylabel("Fine-tuned CLIP", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"results/retrieval_results.png", dpi=150, bbox_inches='tight')
    plt.close()  # close instead of show
    print(f"Results saved to results/retrieval_results.png")





if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    
    # Load both models
    vanilla_model, vanilla_processor, finetuned_model, finetuned_processor, device = load_models()
    
    # Build or load vanilla index
    if os.path.exists("results/vanilla_embeddings.npy"):
        print("Loading saved vanilla index...")
        vanilla_embeddings = np.load("results/vanilla_embeddings.npy")
        with open("results/image_paths.json", "r") as f:
            image_paths = json.load(f)
        with open("results/texts.json", "r") as f:
            texts = json.load(f)
    else:
        print("\nBuilding vanilla CLIP index...")
        vanilla_embeddings, image_paths, texts = build_index(
            vanilla_model, vanilla_processor, "data/dataset.json", device
        )
        np.save("results/vanilla_embeddings.npy", vanilla_embeddings)
        with open("results/image_paths.json", "w") as f:
            json.dump(image_paths, f)
        with open("results/texts.json", "w") as f:
            json.dump(texts, f)

    # Build or load fine-tuned index
    if os.path.exists("results/finetuned_embeddings.npy"):
        print("Loading saved fine-tuned index...")
        finetuned_embeddings = np.load("results/finetuned_embeddings.npy")
    else:
        print("\nBuilding fine-tuned CLIP index...")
        finetuned_embeddings, _, _ = build_index(
            finetuned_model, finetuned_processor, "data/dataset.json", device
        )
        np.save("results/finetuned_embeddings.npy", finetuned_embeddings)

    # Run a query through both models
    query = "bilateral pleural effusion"
    print(f"\nSearching for: '{query}'")
    
    vanilla_results = retrieve(
        query, vanilla_model, vanilla_processor,
        vanilla_embeddings, image_paths, texts, device
    )
    
    finetuned_results = retrieve(
        query, finetuned_model, finetuned_processor,
        finetuned_embeddings, image_paths, texts, device
    )
    
    # Display comparison
    display_results(query, vanilla_results, finetuned_results)
    
    # Print scores
    print("\nVanilla CLIP scores:")
    for r in vanilla_results:
        print(f"  {r['score']:.3f}  {r['image_path']}")
    
    print("\nFine-tuned CLIP scores:")
    for r in finetuned_results:
        print(f"  {r['score']:.3f}  {r['image_path']}")

    # Print fine-tuned report texts
    print("\nFine-tuned report texts:")
    for r in finetuned_results:
        print(f"  Score: {r['score']:.3f}")
        print(f"  Text: {r['text'][:150]}")
        print()