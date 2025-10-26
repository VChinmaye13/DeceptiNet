import argparse, torch, os
from sklearn.metrics import classification_report, roc_auc_score
from src.config import load_config
from src.detector.dataset import get_dataloaders
from src.detector.model import build_model

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    data_root = cfg["paths"]["data_root"]
    img_size = cfg["training"]["img_size"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    model_name = cfg["training"]["model_name"]

    _, _, test_dl = get_dataloaders(data_root, img_size, batch_size, num_workers)

    # -----------------------------
    # ✅ Updated checkpoint logic
    # -----------------------------
    ckpt = args.weights  # user-supplied path
    ckpt_dir = cfg["paths"]["checkpoints"]
    best_ckpt = os.path.join(ckpt_dir, f"detector_{model_name}_best.pt")
    last_ckpt = os.path.join(ckpt_dir, f"detector_{model_name}_last.pt")

    if ckpt:
        # If user provided weights, check they exist
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Provided weights file does not exist: {ckpt}")
        print(f"[INFO] Loading provided checkpoint: {ckpt}")
    else:
        # If none provided, try best first, then last
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
                "Or provide --weights <path_to_file>"
            )

    # Load the checkpoint
    state = torch.load(ckpt, map_location="cpu")
    model = build_model(model_name=model_name).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_probs, all_labels = [], []
    for xb, yb in test_dl:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy().tolist()
        labels = yb.cpu().numpy().tolist()
        all_probs.extend(probs); all_labels.extend(labels)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float('nan')
    print("ROC-AUC:", auc)
    print(classification_report(all_labels, [int(p>=cfg['eval']['threshold']) for p in all_probs], digits=4))

if __name__ == "__main__":
    main()
