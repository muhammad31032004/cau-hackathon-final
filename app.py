"""
WhiteCoat.dev -- Skin Lesion Analysis Demo
===========================================
Task 1: Classification (12 classes, EfficientNetV2-S)
Task 2: Segmentation (binary mask, UNet++ EfficientNet-B4)

AI in Healthcare Hackathon 2026 | Central Asian University
For research and demonstration purposes only. Not for clinical use.

Run: streamlit run app.py
"""

import os
import io
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Drive file IDs
CLASSIFICATION_GDRIVE_ID = "1yc_j-5BlQPE40slVqZZYmPTebxQtn5Wh"
SEGMENTATION_GDRIVE_ID = "1F9GRpXlpUKISoJjnCZhu8HyJNW6mKEnH"

# Local paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "classification.pth")
SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, "segmentation.pth")

# Classification config
CLASSIFICATION_IMG_SIZE = 384
NUM_CLASSES = 12
CLASSIFICATION_MODEL_NAME = "tf_efficientnetv2_s"

# Segmentation config
SEGMENTATION_IMG_SIZE = 256
SEGMENTATION_ENCODER = "efficientnet-b4"

# Class names (hidden by organizers, using domain knowledge)
CLASS_NAMES = {
    0: "Class 0 - Skin Condition",
    1: "Class 1 - Skin Condition",
    2: "Class 2 - Skin Condition",
    3: "Class 3 - Skin Condition",
    4: "Class 4 - Skin Condition",
    5: "Class 5 - Skin Condition",
    6: "Class 6 - Skin Condition",
    7: "Class 7 - Skin Condition",
    8: "Class 8 - Skin Condition",
    9: "Class 9 - Skin Condition",
    10: "Class 10 - Skin Condition",
    11: "Class 11 - Skin Condition",
}


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------
def download_model(gdrive_id, save_path):
    """Download model from Google Drive if not present locally."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        with st.spinner(f"Downloading model... This may take a minute."):
            gdown.download(
                f"https://drive.google.com/uc?id={gdrive_id}",
                save_path, quiet=False,
            )


# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_classification_model():
    download_model(CLASSIFICATION_GDRIVE_ID, CLASSIFICATION_MODEL_PATH)
    checkpoint = torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Handle both checkpoint formats
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg = checkpoint["config"]
        model_name = cfg.get("model_name", CLASSIFICATION_MODEL_NAME)
        num_classes = cfg.get("num_classes", NUM_CLASSES)
        dropout = cfg.get("dropout", 0.3)
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, drop_rate=dropout)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = timm.create_model(CLASSIFICATION_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Plain state_dict
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
        model = timm.create_model(CLASSIFICATION_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)
        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def get_classification_transform(img_size=CLASSIFICATION_IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def classify_image(model, image_np):
    """Classify a single image. Returns (class_id, confidence, all_probabilities)."""
    tfm = get_classification_transform()
    augmented = tfm(image=image_np)
    inp = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(inp)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence, probs


def classify_with_tta(model, image_np):
    """Classify with Test-Time Augmentation (4 views)."""
    tfm = get_classification_transform()
    views = [
        image_np,
        np.fliplr(image_np).copy(),
        np.flipud(image_np).copy(),
        np.fliplr(np.flipud(image_np)).copy(),
    ]
    all_probs = []
    for view in views:
        augmented = tfm(image=view)
        inp = augmented["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(inp)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    predicted_class = int(np.argmax(avg_probs))
    confidence = float(avg_probs[predicted_class])
    return predicted_class, confidence, avg_probs


# ---------------------------------------------------------------------------
# Segmentation model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_segmentation_model():
    download_model(SEGMENTATION_GDRIVE_ID, SEGMENTATION_MODEL_PATH)

    model = smp.UnetPlusPlus(
        encoder_name=SEGMENTATION_ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Extract state dict from various checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith("encoder") or k.startswith("decoder") for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'model.' prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
        cleaned[new_key] = v

    model.load_state_dict(cleaned, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


def get_segmentation_transform(img_size=SEGMENTATION_IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def segment_image(model, image_np, threshold=0.5):
    """Segment a single image. Returns (binary_mask, probability_map)."""
    orig_h, orig_w = image_np.shape[:2]
    tfm = get_segmentation_transform()
    augmented = tfm(image=image_np)
    inp = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(inp)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    prob_resized = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_resized > threshold).astype(np.uint8) * 255
    return binary_mask, prob_resized


def segment_with_tta(model, image_np, threshold=0.5):
    """Segment with Test-Time Augmentation (4 views)."""
    orig_h, orig_w = image_np.shape[:2]
    tfm = get_segmentation_transform()
    preds = []

    views = [
        ("original", image_np),
        ("hflip", np.fliplr(image_np).copy()),
        ("vflip", np.flipud(image_np).copy()),
        ("hvflip", np.fliplr(np.flipud(image_np)).copy()),
    ]

    for name, view in views:
        augmented = tfm(image=view)
        inp = augmented["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
        # Undo transform
        if name == "hflip":
            prob = np.fliplr(prob).copy()
        elif name == "vflip":
            prob = np.flipud(prob).copy()
        elif name == "hvflip":
            prob = np.fliplr(np.flipud(prob)).copy()
        preds.append(prob)

    avg = np.mean(preds, axis=0)
    prob_resized = cv2.resize(avg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_resized > threshold).astype(np.uint8) * 255
    return binary_mask, prob_resized


def create_overlay(image, mask, color=(255, 0, 0), alpha=0.4):
    """Create overlay visualization."""
    overlay = image.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="WhiteCoat.dev - Skin Lesion AI",
        page_icon="🩺",
        layout="wide",
    )

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🩺 WhiteCoat.dev</h1>
        <h3>AI-Powered Skin Lesion Analysis</h3>
        <p style="color: #888;">AI in Healthcare Hackathon 2026 | Central Asian University</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ For research and demonstration purposes only. Not for clinical use.")

    # Task selection
    task = st.selectbox(
        "Select Task",
        ["🔬 Task 1 — Classification (12 Classes)", "🎯 Task 2 — Segmentation (Binary Mask)"],
    )

    st.markdown("---")

    if "Classification" in task:
        run_classification()
    else:
        run_segmentation()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem 0;">
        <p><strong>Team WhiteCoat.dev (#37)</strong> | AI in Healthcare Hackathon 2026</p>
        <p style="color: #c00;">For research and demonstration purposes only. Not for clinical use.</p>
    </div>
    """, unsafe_allow_html=True)


def run_classification():
    st.header("Task 1 — Skin Lesion Classification")
    st.markdown("Upload a skin lesion image to classify it into one of **12 disease categories**.")

    # Sidebar settings
    st.sidebar.header("Classification Settings")
    use_tta = st.sidebar.checkbox("Enable TTA (4 views)", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** EfficientNetV2-S")
    st.sidebar.markdown("**Classes:** 12")
    st.sidebar.markdown("**Input:** 384 x 384")
    st.sidebar.markdown("**Accuracy:** 97.90%")
    st.sidebar.markdown(f"**Device:** `{DEVICE}`")

    # Load model
    model = load_classification_model()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a skin lesion image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="classification_upload",
    )

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.subheader("Uploaded Image")
            st.image(image_rgb, use_container_width=True)

        with st.spinner("Classifying..."):
            if use_tta:
                pred_class, confidence, probs = classify_with_tta(model, image_rgb)
            else:
                pred_class, confidence, probs = classify_image(model, image_rgb)

        with col_result:
            st.subheader("Prediction")
            st.metric("Predicted Class", f"Class {pred_class}")
            st.metric("Confidence", f"{confidence:.2%}")

            if confidence < 0.7:
                st.warning("Low confidence — consider expert review.")

        # Show probability distribution
        st.markdown("---")
        st.subheader("Class Probabilities")

        # Bar chart
        import pandas as pd
        prob_df = pd.DataFrame({
            "Class": [f"Class {i}" for i in range(NUM_CLASSES)],
            "Probability": probs,
        })
        st.bar_chart(prob_df.set_index("Class"))

        # Top 3 predictions
        st.subheader("Top 3 Predictions")
        top3_idx = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(top3_idx, 1):
            st.write(f"**#{rank}** — Class {idx}: {probs[idx]:.4f} ({probs[idx]:.2%})")

    else:
        st.info("Please upload a skin lesion image to begin classification.")


def run_segmentation():
    st.header("Task 2 — Skin Lesion Segmentation")
    st.markdown("Upload a skin lesion image to generate a **binary segmentation mask**.")

    # Sidebar settings
    st.sidebar.header("Segmentation Settings")
    threshold = st.sidebar.slider("Binarization Threshold", 0.1, 0.9, 0.5, 0.05)
    use_tta = st.sidebar.checkbox("Enable TTA (4 views)", value=True, key="seg_tta")
    overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.1, 0.8, 0.4, 0.05)
    overlay_color_name = st.sidebar.selectbox("Overlay Color", ["Red", "Green", "Blue", "Yellow"])
    color_map = {"Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255), "Yellow": (255, 255, 0)}

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** UNet++ (EfficientNet-B4)")
    st.sidebar.markdown("**Input:** 256 x 256")
    st.sidebar.markdown("**Metric:** Mean IoU")
    st.sidebar.markdown(f"**Device:** `{DEVICE}`")

    # Load model
    model = load_segmentation_model()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a skin lesion image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="segmentation_upload",
    )

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Running segmentation..."):
            if use_tta:
                binary_mask, prob_map = segment_with_tta(model, image_rgb, threshold)
            else:
                binary_mask, prob_map = segment_image(model, image_rgb, threshold)

        overlay = create_overlay(
            image_rgb, binary_mask,
            color=color_map[overlay_color_name],
            alpha=overlay_alpha,
        )

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original")
            st.image(image_rgb, use_container_width=True)
        with col2:
            st.subheader("Predicted Mask")
            st.image(binary_mask, use_container_width=True, clamp=True)
        with col3:
            st.subheader("Overlay")
            st.image(overlay, use_container_width=True)

        # Heatmap and stats
        st.markdown("---")
        col4, col5 = st.columns(2)

        with col4:
            st.subheader("Probability Heatmap")
            heatmap = cv2.applyColorMap(
                (prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            st.image(heatmap_rgb, use_container_width=True)

        with col5:
            st.subheader("Statistics")
            fg_ratio = (binary_mask > 127).sum() / binary_mask.size
            st.metric("Lesion Area", f"{fg_ratio:.2%}")
            st.metric("Image Size", f"{image_rgb.shape[1]} x {image_rgb.shape[0]}")
            st.metric("Mean Probability", f"{prob_map.mean():.4f}")
            st.metric("Max Probability", f"{prob_map.max():.4f}")

        # Download mask
        st.markdown("---")
        mask_pil = Image.fromarray(binary_mask)
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            label="Download Predicted Mask (PNG)",
            data=buf.getvalue(),
            file_name=f"mask_{uploaded_file.name.split('.')[0]}.png",
            mime="image/png",
        )

    else:
        st.info("Please upload a skin lesion image to begin segmentation.")


if __name__ == "__main__":
    main()
