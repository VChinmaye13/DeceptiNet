#!/usr/bin/env python3
import sys
import os
import argparse
from PIL import Image

# Ensure project root is on Python path so "src" imports work regardless of cwd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import gradio as gr
from torchvision import transforms

from src.config import load_config
from src.detector.model import build_model

def build_pipeline(cfg, weights_path=None):
    # Choose device robustly
    device_cfg = cfg.get("project", {}).get("device", "cpu")
    device = "cuda" if (device_cfg == "cuda" and torch.cuda.is_available()) else "cpu"

    model_name = cfg["training"]["model_name"]
    ckpt_dir = cfg["paths"]["checkpoints"]
    best_ckpt = os.path.join(ckpt_dir, f"detector_{model_name}_best.pt")
    last_ckpt = os.path.join(ckpt_dir, f"detector_{model_name}_last.pt")

    # Decide which checkpoint to use
    ckpt = None
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Provided weights file does not exist: {weights_path}")
        ckpt = weights_path
        print(f"[INFO] Loading provided checkpoint: {ckpt}")
    else:
        if os.path.exists(best_ckpt):
            ckpt = best_ckpt
            print(f"[INFO] Loading best checkpoint: {ckpt}")
        elif os.path.exists(last_ckpt):
            ckpt = last_ckpt
            print(f"[WARNING] Best checkpoint not found; loading last checkpoint instead: {ckpt}")
        else:
            raise FileNotFoundError(
                "No checkpoint found. Expected one of:\n"
                f"  {best_ckpt}\n  {last_ckpt}\n"
                "Train a model first or provide --weights <path_to_pt>."
            )

    # Load model
    state = torch.load(ckpt, map_location="cpu")
    model = build_model(model_name=model_name).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Image transform (match dataset transforms)
    img_size = cfg["training"]["img_size"]
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, device, tfm

def predict_image(model, img: Image.Image, device, tfm):
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob_fake = float(torch.softmax(logits, dim=1)[0, 1].cpu().numpy())
    return prob_fake

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Optional: path to .pt weights")
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        model, device, tfm = build_pipeline(cfg, args.weights)
    except FileNotFoundError as e:
        # Clear, actionable message for the user
        print("ERROR:", e)
        print("\nTip: Run training first to produce checkpoints, or pass --weights <path_to_weights.pt>.")
        raise

    def infer(img):
        prob_fake = predict_image(model, img, device, tfm)
        label = "Fake" if prob_fake >= cfg["eval"]["threshold"] else "Real"
        scores = {"Real": float(1.0 - prob_fake), "Fake": float(prob_fake)}
        return scores, f"{label} (p_fake={prob_fake:.4f})"

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Image(type="pil", label="Upload face image"),
        outputs=[gr.Label(num_top_classes=2, label="Scores"), gr.Textbox(label="Prediction")],
        title="DeceptiNet Deepfake Detector",
        description="Upload a face image and the model will predict Real vs Fake."
    )

    demo.launch()

if __name__ == "__main__":
    main()
