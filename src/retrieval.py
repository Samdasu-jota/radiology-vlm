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

    print("Loading negation-aware CLIP...")
    negation_model = CLIPModel.from_pretrained("model/negation_aware_clip")
    negation_processor = CLIPProcessor.from_pretrained("model/negation_aware_clip")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU) ✅")
    else:
        device = torch.device("cpu")

    vanilla_model = vanilla_model.to(device)
    finetuned_model = finetuned_model.to(device)
    negation_model = negation_model.to(device)

    vanilla_model.eval()
    finetuned_model.eval()
    negation_model.eval()

    return (vanilla_model, vanilla_processor,
            finetuned_model, finetuned_processor,
            negation_model, negation_processor,
            device)


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

            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            vision_output = model.vision_model(pixel_values=pixel_values)
            image_embedding = vision_output.pooler_output
            image_embedding = model.visual_projection(image_embedding)
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
    model.eval()
    with torch.no_grad():
        inputs = processor(
            text=query_text,
            return_tensors="pt",
            padding=True
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        text_output = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        query_embedding = text_output.pooler_output
        query_embedding = model.text_projection(query_embedding)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy()

    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "image_path": image_paths[idx],
            "text": texts[idx],
            "score": float(similarities[idx])
        })

    return results


def display_results(query_vanilla, query_finetuned, query_negation,
                    vanilla_results, finetuned_results, negation_results):
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("Retrieval Comparison: Vanilla vs Fine-tuned vs Negation-Aware",
                 fontsize=14, fontweight='bold')

    # Row 1 — vanilla CLIP
    for i, result in enumerate(vanilla_results):
        img = Image.open(result["image_path"]).convert("RGB")
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"Score: {result['score']:.3f}", fontsize=9)
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel(f"Vanilla\n'{query_vanilla}'", fontsize=10, fontweight='bold')

    # Row 2 — fine-tuned CLIP
    for i, result in enumerate(finetuned_results):
        img = Image.open(result["image_path"]).convert("RGB")
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(f"Score: {result['score']:.3f}", fontsize=9)
        axes[1, i].axis("off")
    axes[1, 0].set_ylabel(f"Fine-tuned\n'{query_finetuned}'", fontsize=10, fontweight='bold')

    # Row 3 — negation-aware CLIP
    for i, result in enumerate(negation_results):
        img = Image.open(result["image_path"]).convert("RGB")
        axes[2, i].imshow(img, cmap="gray")
        axes[2, i].set_title(f"Score: {result['score']:.3f}", fontsize=9)
        axes[2, i].axis("off")
    axes[2, 0].set_ylabel(f"Negation-Aware\n'{query_negation}'", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("results/retrieval_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Results saved to results/retrieval_comparison.png")


def print_results(label, results):
    print(f"\n{label}:")
    for r in results:
        print(f"  {r['score']:.3f}  {r['text'][:120]}")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    # Load all three models
    (vanilla_model, vanilla_processor,
     finetuned_model, finetuned_processor,
     negation_model, negation_processor,
     device) = load_models()

    # Load or build vanilla index
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

    # Load or build fine-tuned index
    if os.path.exists("results/finetuned_embeddings.npy"):
        print("Loading saved fine-tuned index...")
        finetuned_embeddings = np.load("results/finetuned_embeddings.npy")
    else:
        print("\nBuilding fine-tuned CLIP index...")
        finetuned_embeddings, _, _ = build_index(
            finetuned_model, finetuned_processor, "data/dataset.json", device
        )
        np.save("results/finetuned_embeddings.npy", finetuned_embeddings)

    # Load or build negation-aware index
    if os.path.exists("results/negation_embeddings.npy"):
        print("Loading saved negation index...")
        negation_embeddings = np.load("results/negation_embeddings.npy")
    else:
        print("\nBuilding negation-aware index...")
        negation_embeddings, _, _ = build_index(
            negation_model, negation_processor, "data/dataset.json", device
        )
        np.save("results/negation_embeddings.npy", negation_embeddings)

    # Queries — note structured format for negation model
    query_vanilla   = "bilateral pleural effusion"
    query_finetuned = "bilateral pleural effusion"
    query_negation  = "present: pleural effusion"

    print(f"\nSearching...")

    vanilla_results = retrieve(
        query_vanilla, vanilla_model, vanilla_processor,
        vanilla_embeddings, image_paths, texts, device
    )

    finetuned_results = retrieve(
        query_finetuned, finetuned_model, finetuned_processor,
        finetuned_embeddings, image_paths, texts, device
    )

    negation_results = retrieve(
        query_negation, negation_model, negation_processor,
        negation_embeddings, image_paths, texts, device
    )

    # Display visual comparison
    display_results(
        query_vanilla, query_finetuned, query_negation,
        vanilla_results, finetuned_results, negation_results
    )

    # Print report texts for each model
    print_results("Vanilla CLIP", vanilla_results)
    print_results("Fine-tuned CLIP", finetuned_results)
    print_results("Negation-Aware CLIP", negation_results)