import streamlit as st
import torch
import json
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from src.retrieval import build_index, retrieve

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Radiology Retrieval",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Radiology Vision-Language Retrieval")
st.markdown("""
Fine-tuned CLIP for chest X-ray retrieval.
Compare vanilla, fine-tuned, and negation-aware models side by side.
""")

# ── Load models (cached so they only load once) ───────────────────────────────
@st.cache_resource
def load_all_models():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    models = {}

    with st.spinner("Loading Vanilla CLIP..."):
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        m.eval()
        models["vanilla"] = (m, p)

    with st.spinner("Loading Fine-tuned CLIP..."):
        m = CLIPModel.from_pretrained("model/fine_tuned_clip").to(device)
        p = CLIPProcessor.from_pretrained("model/fine_tuned_clip")
        m.eval()
        models["finetuned"] = (m, p)

    with st.spinner("Loading Negation-Aware CLIP..."):
        m = CLIPModel.from_pretrained("model/negation_aware_clip").to(device)
        p = CLIPProcessor.from_pretrained("model/negation_aware_clip")
        m.eval()
        models["negation"] = (m, p)

    return models, device


@st.cache_resource
def load_all_indexes(_models, _device):
    indexes = {}

    with st.spinner("Loading search indexes..."):
        image_paths = json.load(open("results/image_paths.json"))
        texts = json.load(open("results/texts.json"))

        indexes["vanilla"] = (
            np.load("results/vanilla_embeddings.npy"),
            image_paths, texts
        )
        indexes["finetuned"] = (
            np.load("results/finetuned_embeddings.npy"),
            image_paths, texts
        )
        indexes["negation"] = (
            np.load("results/negation_embeddings.npy"),
            image_paths, texts
        )

    return indexes


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.radio(
        "Model",
        options=["vanilla", "finetuned", "negation"],
        format_func=lambda x: {
            "vanilla": "Vanilla CLIP",
            "finetuned": "Fine-tuned CLIP",
            "negation": "Negation-Aware CLIP ⭐"
        }[x],
        index=2
    )

    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### 💡 Example queries")
    st.markdown("""
    **Negation-Aware model:**
    - `present: pleural effusion`
    - `present: cardiomegaly`
    - `absent: pneumothorax`

    **Vanilla / Fine-tuned:**
    - `bilateral pleural effusion`
    - `enlarged heart`
    - `normal chest x-ray`
    """)

    st.markdown("---")
    st.markdown("### 📊 Model comparison")
    st.markdown("""
    | Model | Negation? |
    |-------|-----------|
    | Vanilla | ❌ |
    | Fine-tuned | ⚠️ |
    | Negation-Aware | ✅ |
    """)


# ── Main search interface ─────────────────────────────────────────────────────
query = st.text_input(
    "Search query",
    value="present: pleural effusion",
    placeholder="e.g. present: pleural effusion"
)

search_clicked = st.button("🔍 Search", type="primary")

# ── Load models and indexes ───────────────────────────────────────────────────
models, device = load_all_models()
indexes = load_all_indexes(models, device)

# ── Run search ────────────────────────────────────────────────────────────────
if search_clicked and query:
    model, processor = models[model_choice]
    embeddings, image_paths, texts = indexes[model_choice]

    with st.spinner("Searching..."):
        results = retrieve(
            query, model, processor,
            embeddings, image_paths, texts,
            device, top_k=top_k
        )

    st.markdown(f"### Results for: `{query}`")
    st.markdown(f"*Model: {model_choice}*")
    st.markdown("---")

    # Display results in columns
    cols = st.columns(min(top_k, 5))
    for i, result in enumerate(results[:5]):
        with cols[i]:
            img = Image.open(result["image_path"]).convert("RGB")
            st.image(img, use_container_width=True)
            st.metric("Score", f"{result['score']:.3f}")
            with st.expander("Report"):
                st.write(result["text"])

    # Second row if top_k > 5
    if top_k > 5 and len(results) > 5:
        cols2 = st.columns(min(top_k - 5, 5))
        for i, result in enumerate(results[5:]):
            with cols2[i]:
                img = Image.open(result["image_path"]).convert("RGB")
                st.image(img, use_container_width=True)
                st.metric("Score", f"{result['score']:.3f}")
                with st.expander("Report"):
                    st.write(result["text"])
