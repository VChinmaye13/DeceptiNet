import argparse, os, torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.config import load_config
from src.utils.common import set_seed, ensure_dir, to_device
from src.detector.dataset import get_dataloaders
from src.detector.model import build_model

def train_one_epoch(model, dl, criterion, optimizer, device, use_amp):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    running_loss, all_preds, all_labels = 0.0, [], []
    for xb, yb in tqdm(dl, desc="Train", ncols=80):
        xb, yb = to_device(xb, device), to_device(yb, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        labels = yb.detach().cpu().numpy().tolist()
        all_preds.extend(preds); all_labels.extend(labels)

    epoch_loss = running_loss / len(dl.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, acc, f1

@torch.no_grad()
def validate(model, dl, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    for xb, yb in tqdm(dl, desc="Val", ncols=80):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        labels = yb.cpu().numpy().tolist()
        all_preds.extend(preds); all_labels.extend(labels)
    epoch_loss = running_loss / len(dl.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"

    img_size = cfg["training"]["img_size"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    model_name = cfg["training"]["model_name"]
    use_amp = bool(cfg["training"].get("mixed_precision", True))

    data_root = cfg["paths"]["data_root"]
    ckpt_dir = cfg["paths"]["checkpoints"]
    ensure_dir(ckpt_dir)

    train_dl, val_dl, _ = get_dataloaders(data_root, img_size, batch_size, num_workers)

    model = build_model(model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_f1, best_path = 0.0, None
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_dl, criterion, optimizer, device, use_amp)
        vl_loss, vl_acc, vl_f1 = validate(model, val_dl, criterion, device)
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | val_loss={vl_loss:.4f} acc={vl_acc:.4f} f1={vl_f1:.4f}")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_path = os.path.join(ckpt_dir, f"detector_{model_name}_best.pt")
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, best_path)
            print("Saved:", best_path)

    if best_path is None:
        # Always save a last checkpoint
        best_path = os.path.join(ckpt_dir, f"detector_{model_name}_last.pt")
        torch.save({"model_state": model.state_dict(), "cfg": cfg}, best_path)
        print("Saved last:", best_path)

if __name__ == "__main__":
    main()
