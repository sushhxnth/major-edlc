import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch

from src.model import EnhancedSSAPeriocularNet
from src.pipeline import detect_and_crop_periocular, init_face_landmarker, run_model_inference


def enroll_user(image_paths: list[str], name: str, person_id: str, gender: str, role: str, notes: str):
    base_dir = Path(__file__).resolve().parent
    ckpt_path = base_dir / "checkpoints" / "ssa_periocular_best.pt"
    reg_path = base_dir / "gallery" / "registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = EnhancedSSAPeriocularNet(
        num_identities=ckpt["num_identities"],
        backbone_name=ckpt.get("backbone", "resnet34"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    detector = init_face_landmarker(base_dir / "checkpoints")
    embeddings = []

    for img_str in image_paths:
        bgr = cv2.imread(img_str)
        if bgr is None:
            print(f"⚠️ Skipping unreadable image: {img_str}")
            continue
        res = detect_and_crop_periocular(bgr, detector, use_clahe=True)
        for side, edata in res["eyes"].items():
            out = run_model_inference(model, edata["processed_rgb"], device)
            embeddings.append(out["embedding"])

    if not embeddings:
        raise RuntimeError("No valid periocular regions extracted from the supplied images.")

    mean_template = np.mean(embeddings, axis=0)
    mean_template /= max(np.linalg.norm(mean_template), 1e-12)

    registry = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else []
    registry = [r for r in registry if r["person_id"].lower() != person_id.lower()]
    registry.append({
        "person_id": person_id,
        "name": name,
        "gender": gender,
        "role": role,
        "notes": notes,
        "num_crops_enrolled": len(embeddings),
        "embedding": mean_template.tolist(),
    })

    reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f" Enrolled {name} ({person_id}) using {len(embeddings)} periocular crops into {reg_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, help="1 or more photos of your face")
    parser.add_argument("--name", type=str, default="B. Sushanth Rao")
    parser.add_argument("--id", type=str, default="CSE-2027-SUSHANTH")
    parser.add_argument("--gender", type=str, default="Male", choices=["Male", "Female"])
    parser.add_argument("--role", type=str, default="B.Tech CSE Final Year - Project Lead")
    parser.add_argument("--notes", type=str, default="Matrusri Engineering College")
    args = parser.parse_args()

    enroll_user(args.images, args.name, args.id, args.gender, args.role, args.notes)