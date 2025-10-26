"""
Minimal face extraction utility (optional).

Usage:
  python -m src.preprocessing.face_extract --in_path data/raw/sample.jpg --out_dir data/processed/train/real
"""
import argparse, os
from PIL import Image
from facenet_pytorch import MTCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    mtcnn = MTCNN(image_size=args.size, margin=20, post_process=True)
    img = Image.open(args.in_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        print("No face detected.")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, os.path.basename(args.in_path))
    face_img = Image.fromarray((face.permute(1,2,0).numpy()*255).astype('uint8'))
    face_img.save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
