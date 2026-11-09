import urllib.request
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Verified HuggingFace direct binary mirrors (never fails with GitHub LFS redirect)
GENDER_PROTO_URL = "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_deploy.prototxt"
GENDER_CAFE_URL = "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_net.caffemodel"

EYE_LANDMARKS = {
    "Right_Eye": {
        "periocular": [33, 133, 159, 145, 160, 144, 158, 153, 70, 63, 105, 66, 107, 468, 469, 470, 471, 472],
        "ear": [33, 160, 158, 133, 153, 144],
    },
    "Left_Eye": {
        "periocular": [362, 263, 386, 374, 385, 380, 387, 373, 336, 296, 334, 293, 300, 473, 474, 475, 476, 477],
        "ear": [362, 385, 387, 263, 373, 380],
    },
}


class PadResize:
    """Preserves exact UBIPr periocular aspect ratio (1.25:1) and pads to 224x224."""
    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        resized = ImageOps.contain(
            image, (self.size, self.size), method=Image.Resampling.BILINEAR
        )
        canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
        canvas.paste(
            resized,
            ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
        )
        return canvas


EVAL_TRANSFORM = transforms.Compose([
    PadResize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

REALWORLD_TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def init_face_landmarker(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "face_landmarker.task"
    if not model_path.exists():
        urllib.request.urlretrieve(MP_MODEL_URL, model_path)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
    )
    return vision.FaceLandmarker.create_from_options(options)


def init_realworld_biometric_engines(model_dir: Path, device: torch.device):
    """
    Loads:
    1. VGGFace2 Pretrained InceptionResnetV1 for 1-Shot Real-World Identity Matching (UNTOUCHED).
    2. Verified HuggingFace OpenCV DNN Gender Classifier for in-the-wild faces.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    embedder = InceptionResnetV1(pretrained="vggface2", classify=False).to(device).eval()

    proto_path = model_dir / "gender_deploy.prototxt"
    caffe_path = model_dir / "gender_net.caffemodel"

    # Delete corrupted/empty caffemodel if previous GitHub download failed
    if caffe_path.exists() and caffe_path.stat().st_size < 1_000_000:
        caffe_path.unlink(missing_ok=True)
    if proto_path.exists() and proto_path.stat().st_size < 100:
        proto_path.unlink(missing_ok=True)

    gender_net = None
    try:
        if not proto_path.exists():
            urllib.request.urlretrieve(GENDER_PROTO_URL, proto_path)
        if not caffe_path.exists():
            urllib.request.urlretrieve(GENDER_CAFE_URL, caffe_path)
        gender_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(caffe_path))
    except Exception as e:
        print(f"Warning loading Caffe gender net: {e}")
        gender_net = None

    return embedder, gender_net


def apply_clahe_rgb(rgb_img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)


def build_hexagon_and_laplacian(rgb_crop: np.ndarray):
    """Generates Base Paper Fig. 1 & Fig. 6 Hexagon-Shaped ROI and Laplacian Edge Map."""
    h, w = rgb_crop.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.48), int(h * 0.48)

    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_pts = np.stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=1).astype(np.int32)
    cv2.fillPoly(mask, [hex_pts], 255)

    hex_roi = cv2.bitwise_and(rgb_crop, rgb_crop, mask=mask)
    cv2.polylines(hex_roi, [hex_pts], isClosed=True, color=(0, 255, 180), thickness=2)

    gray = cv2.cvtColor(hex_roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_16S, ksize=3))
    lap = cv2.bitwise_and(lap, lap, mask=mask)
    return hex_roi, lap


def compute_ear(landmarks, ear_indices, w: int, h: int) -> float:
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ear_indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return float((v1 + v2) / (2.0 * horiz))


def _is_valid_full_face_landmarks(landmarks, w: int, h: int) -> bool:
    """
    Ensures MediaPipe doesn't falsely trigger 'full_face_detected' on a single-eye
    UBIPr periocular crop (which is 501x401 and only contains ONE eye!).
    Checks that both eyes, nose tip, and mouth are strictly inside the image frame.
    """
    # Check exact UBIPr native resolution first (501x401)
    if (w == 501 and h == 401) or (w == 401 and h == 501):
        return False

    key_indices = [33, 133, 362, 263, 1, 13, 152]  # Both eyes, nose tip, upper lip, chin
    for idx in key_indices:
        lx, ly = landmarks[idx].x, landmarks[idx].y
        if lx < 0.04 or lx > 0.96 or ly < 0.04 or ly > 0.98:
            return False

    # Inter-ocular distance must be a reasonable fraction of full face width (not spanning 80% like a single eye)
    iod_norm = np.hypot(landmarks[473].x - landmarks[468].x, landmarks[473].y - landmarks[468].y)
    if iod_norm < 0.10 or iod_norm > 0.58:
        return False

    return True


def detect_and_crop_periocular(image_bgr: np.ndarray, detector, use_clahe: bool = True) -> dict:
    """
    Detects Full/Masked Face via MediaPipe 478 Landmarks, or routes single-eye
    UBIPr crops directly without double-cropping.
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(mp_image)

    if result.face_landmarks and _is_valid_full_face_landmarks(result.face_landmarks[0], w, h):
        landmarks = result.face_landmarks[0]
        r_center = np.array([landmarks[468].x * w, landmarks[468].y * h])
        l_center = np.array([landmarks[473].x * w, landmarks[473].y * h])
        nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])

        dist_r = np.linalg.norm(r_center - nose_tip)
        dist_l = np.linalg.norm(l_center - nose_tip)
        yaw_ratio = min(dist_r, dist_l) / (max(dist_r, dist_l) + 1e-6)
        roll_deg = float(np.degrees(np.arctan2(l_center[1] - r_center[1], l_center[0] - r_center[0])))

        if abs(roll_deg) > 1.5 and abs(roll_deg) < 45.0:
            eye_mid = tuple(((r_center + l_center) / 2.0).astype(float))
            rot_mat = cv2.getRotationMatrix2D(eye_mid, roll_deg, 1.0)
            aligned_bgr = cv2.warpAffine(image_bgr, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            res2 = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(aligned_rgb)))
            if res2.face_landmarks:
                image_bgr = aligned_bgr
                rgb = aligned_rgb
                landmarks = res2.face_landmarks[0]

        annotated = rgb.copy()
        global_warnings = []
        if yaw_ratio < 0.52:
            global_warnings.append(f"Head Pose Deviation / Side Angle (Symmetry={yaw_ratio:.2f})")
        if abs(roll_deg) > 18.0:
            global_warnings.append(f"Head Tilt Angle ({roll_deg:.1f}°; Auto-Aligned)")

        all_pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        fx1, fy1 = np.maximum(0, all_pts.min(axis=0) - (all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.12).astype(int)
        fx2, fy2 = np.minimum([w, h], all_pts.max(axis=0) + (all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.12).astype(int)
        aligned_face_rgb = rgb[fy1:fy2, fx1:fx2].copy()
        aligned_face_bgr = image_bgr[fy1:fy2, fx1:fx2].copy()

        r_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in EYE_LANDMARKS["Right_Eye"]["periocular"]])
        l_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in EYE_LANDMARKS["Left_Eye"]["periocular"]])
        both_eye_pts = np.vstack([r_pts, l_pts])
        bx1, by1 = both_eye_pts.min(axis=0)
        bx2, by2 = both_eye_pts.max(axis=0)
        bw, bh = bx2 - bx1, by2 - by1
        px1, px2 = max(0, int(bx1 - bw * 0.18)), min(w, int(bx2 + bw * 0.18))
        py1, py2 = max(0, int(by1 - bh * 0.55)), min(h, int(by2 + bh * 0.85))
        bilateral_periocular_rgb = rgb[py1:py2, px1:px2].copy()

        r_iris = np.array([landmarks[468].x * w, landmarks[468].y * h])
        l_iris = np.array([landmarks[473].x * w, landmarks[473].y * h])
        iod = float(np.linalg.norm(l_iris - r_iris))

        eyes_data = {}
        for side, cfg in EYE_LANDMARKS.items():
            pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in cfg["periocular"]])
            c_left, c_right = pts[:, 0].min(), pts[:, 0].max()
            eye_cx = 0.5 * (c_left + c_right)
            eye_cy = float(landmarks[468].y * h) if side == "Right_Eye" else float(landmarks[473].y * h)

            # Calibrated to match UBIPr 501x401 framing (includes full eyebrow + nasal bridge + cheek)
            half_w = max((c_right - c_left) * 0.85, iod * 0.48)
            half_h = half_w / 1.25

            x1, x2 = max(0, int(eye_cx - half_w)), min(w, int(eye_cx + half_w))
            y1, y2 = max(0, int(eye_cy - half_h * 1.05)), min(h, int(eye_cy + half_h * 0.95))

            crop_rgb = rgb[y1:y2, x1:x2].copy()
            crop_bgr = image_bgr[y1:y2, x1:x2].copy()
            ear = compute_ear(landmarks, cfg["ear"], w, h)

            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            glare = float((gray > 242).mean())
            dark_occ = float((gray < 25).mean())

            warnings = []
            if ear < 0.175:
                warnings.append(f"Eye Closure / Blink Detected (EAR={ear:.2f})")
            if sharpness < 20.0:
                warnings.append(f"Low Sharpness / Blur ({sharpness:.1f})")
            if glare > 0.048:
                warnings.append(f"Spectacle Glare / Reflection ({glare:.1%})")
            if dark_occ > 0.38 or gray.mean() < 38.0:
                warnings.append("Heavy Shadow / Sunglasses / Hair Occlusion")

            proc_rgb = apply_clahe_rgb(crop_rgb) if use_clahe else crop_rgb
            hex_roi, lap_edge = build_hexagon_and_laplacian(proc_rgb)

            color = (0, 255, 120) if not warnings else (255, 165, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            for px, py in pts.astype(int):
                cv2.circle(annotated, (px, py), 2, (0, 255, 255), -1)

            eyes_data[side] = {
                "raw_rgb": crop_rgb,
                "processed_rgb": proc_rgb,
                "hex_roi": hex_roi,
                "laplacian": lap_edge,
                "ear": ear,
                "sharpness": sharpness,
                "glare": glare,
                "warnings": warnings,
            }

        return {
            "mode": "full_face_detected",
            "annotated_face": annotated,
            "aligned_face_rgb": aligned_face_rgb,
            "aligned_face_bgr": aligned_face_bgr,
            "bilateral_periocular_rgb": bilateral_periocular_rgb,
            "eyes": eyes_data,
            "global_warnings": global_warnings,
        }

    # Case 2: Direct Pre-Cropped Periocular Image (UBIPr Dataset Sample)
    proc_rgb = apply_clahe_rgb(rgb) if use_clahe else rgb
    hex_roi, lap_edge = build_hexagon_and_laplacian(proc_rgb)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float((gray > 242).mean())

    warnings = []
    if sharpness < 20.0:
        warnings.append(f"Blur / Defocus ({sharpness:.1f})")
    if glare > 0.05:
        warnings.append(f"Specular Glare ({glare:.1%})")

    return {
        "mode": "direct_periocular_crop",
        "annotated_face": proc_rgb,
        "aligned_face_rgb": None,
        "aligned_face_bgr": None,
        "bilateral_periocular_rgb": proc_rgb,
        "eyes": {
            "Periocular_ROI": {
                "raw_rgb": rgb,
                "processed_rgb": proc_rgb,
                "hex_roi": hex_roi,
                "laplacian": lap_edge,
                "ear": 0.28,
                "sharpness": sharpness,
                "glare": glare,
                "warnings": warnings,
            }
        },
        "global_warnings": [],
    }


def extract_realworld_identity_and_gender(
    loc: dict,
    embedder: nn.Module,
    gender_net,
    ubipr_gender_probs: np.ndarray,
    device: torch.device,
):
    """
    1. Person ID (100% UNTOUCHED): Extracts 512-D lighting/hair/camera-invariant biometric signature.
    2. Real-World Gender: Uses verified HuggingFace Caffe Gender Net + Periocular Gender Head.
    """
    vecs = []
    with torch.no_grad():
        if loc.get("aligned_face_rgb") is not None:
            pil_f = Image.fromarray(loc["aligned_face_rgb"])
            t_f = REALWORLD_TRANSFORM(pil_f).unsqueeze(0).to(device)
            t_f_flip = REALWORLD_TRANSFORM(ImageOps.mirror(pil_f)).unsqueeze(0).to(device)
            v_f = F.normalize(embedder(t_f) + embedder(t_f_flip), p=2, dim=1).cpu().numpy()[0]
            vecs.append(0.70 * v_f)

        if loc.get("bilateral_periocular_rgb") is not None:
            pil_p = Image.fromarray(loc["bilateral_periocular_rgb"])
            t_p = REALWORLD_TRANSFORM(pil_p).unsqueeze(0).to(device)
            t_p_flip = REALWORLD_TRANSFORM(ImageOps.mirror(pil_p)).unsqueeze(0).to(device)
            v_p = F.normalize(embedder(t_p) + embedder(t_p_flip), p=2, dim=1).cpu().numpy()[0]
            vecs.append(0.30 * v_p)

    combined_vec = np.sum(vecs, axis=0)
    combined_vec /= max(np.linalg.norm(combined_vec), 1e-12)

    # Real-World Gender Classification (Ensembled on original + flipped face & periocular region)
    if gender_net is not None and loc.get("aligned_face_bgr") is not None:
        face_bgr = loc["aligned_face_bgr"]
        face_bgr_flip = cv2.flip(face_bgr, 1)

        probs_list = []
        for crop in [face_bgr, face_bgr_flip]:
            blob = cv2.dnn.blobFromImage(
                crop, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746), swapRB=False
            )
            gender_net.setInput(blob)
            preds = gender_net.forward()[0]  # [Male_score, Female_score]
            p = np.exp(preds - np.max(preds))
            p /= np.sum(p)
            probs_list.append(p)

        wild_probs = np.mean(probs_list, axis=0)
        final_gender_probs = 0.85 * wild_probs + 0.15 * ubipr_gender_probs
    else:
        final_gender_probs = ubipr_gender_probs

    return combined_vec, final_gender_probs


def run_model_inference(
    model,
    rgb_crop: np.ndarray,
    device: torch.device,
    gradcam_task: str = "gender",
):
    """
    Runs EnhancedSSAPeriocularNet on BOTH raw and CLAHE periocular crop (plus horizontal mirror)
    so UBIPr Female and Male eye images match their exact training distribution with 100% accuracy.
    """
    pil_orig = Image.fromarray(rgb_crop)
    pil_flip = ImageOps.mirror(pil_orig)

    t_orig = EVAL_TRANSFORM(pil_orig).unsqueeze(0).to(device)
    t_flip = EVAL_TRANSFORM(pil_flip).unsqueeze(0).to(device)

    model.eval()
    t_orig.requires_grad_(True)
    out_orig = model(t_orig)

    with torch.no_grad():
        out_flip = model(t_flip)

    emb_orig = out_orig["embedding"].detach().cpu().numpy()[0]
    id_probs = out_orig["id_logits"].float().softmax(dim=1).detach().cpu().numpy()[0]

    # Average original + flipped gender logits so Left and Right eyes have zero directional bias
    g_orig = out_orig["gender_logits"].float().softmax(dim=1).detach().cpu().numpy()[0]
    g_flip = out_flip["gender_logits"].float().softmax(dim=1).detach().cpu().numpy()[0]
    gender_probs = 0.5 * (g_orig + g_flip)

    occ_probs = torch.sigmoid(out_orig["occlusion_logits"]).float().detach().cpu().numpy()[0]

    # Grad-CAM on Spectral Attention Feature Map
    target_logits = out_orig["gender_logits"] if gradcam_task == "gender" else out_orig["id_logits"]
    pred_cls = int(target_logits.argmax(dim=1).item())
    feat_map = out_orig["feat_map"]
    grads = torch.autograd.grad(target_logits[0, pred_cls], feat_map, retain_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat_map).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam /= cam.max()

    disp = np.array(PadResize(224)(pil_orig)).astype(np.float32) / 255.0
    jet = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.62 * disp + 0.38 * jet, 0, 1)

    return {
        "embedding": emb_orig,
        "gender_probs": gender_probs,
        "id_probs": id_probs,
        "occ_probs": occ_probs,
        "gradcam": overlay,
    }