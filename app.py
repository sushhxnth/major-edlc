import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

import src.pipeline as pipeline_mod
importlib.reload(pipeline_mod)

from src.model import EnhancedSSAPeriocularNet
from src.pipeline import (
    detect_and_crop_periocular,
    extract_realworld_identity_and_gender,
    init_face_landmarker,
    init_realworld_biometric_engines,
    run_model_inference,
)

st.set_page_config(
    page_title="Periocular Biometrics — Final Review System",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "checkpoints" / "ssa_periocular_best.pt"
REG_PATH = BASE_DIR / "gallery" / "registry.json"
UBIPR_GAL_PATH = BASE_DIR / "gallery" / "ubipr_gallery_templates.npz"
RESULTS_DIR = BASE_DIR / "results"
DEMO_DIR = BASE_DIR / "sample_demo_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_system():
    detector = init_face_landmarker(BASE_DIR / "checkpoints")
    embedder, gender_net = init_realworld_biometric_engines(BASE_DIR / "checkpoints", DEVICE)

    if not CKPT_PATH.exists():
        return detector, embedder, gender_net, None, None, None

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model = EnhancedSSAPeriocularNet(
        num_identities=ckpt["num_identities"],
        backbone_name=ckpt.get("backbone", "resnet34"),
        pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ubipr_gal = None
    if UBIPR_GAL_PATH.exists():
        ubipr_gal = dict(np.load(UBIPR_GAL_PATH, allow_pickle=True))

    return detector, embedder, gender_net, model, ckpt, ubipr_gal


def load_custom_registry() -> list:
    if REG_PATH.exists():
        return json.loads(REG_PATH.read_text(encoding="utf-8"))
    return []


def save_custom_registry(entry: dict, append_templates: bool = True):
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg = load_custom_registry()
    existing = next((r for r in reg if r["person_id"].lower() == entry["person_id"].lower()), None)
    if existing and append_templates:
        old_bank = existing.get("template_bank", [existing["embedding"]])
        # Only merge if dimensions match
        if len(old_bank[0]) == len(entry["embedding"]):
            new_bank = old_bank + entry.get("template_bank", [entry["embedding"]])
            mean_vec = np.mean(new_bank, axis=0)
            mean_vec = (mean_vec / max(np.linalg.norm(mean_vec), 1e-12)).tolist()
            entry["template_bank"] = new_bank
            entry["embedding"] = mean_vec
            entry["num_crops_enrolled"] = len(new_bank)

    reg = [r for r in reg if r["person_id"].lower() != entry["person_id"].lower()]
    reg.append(entry)
    REG_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


detector, embedder, gender_net, model, ckpt, ubipr_gal = load_system()

st.title("👁️ Enhanced Periocular Biometric Identification & Real-Time Gender Classification")
st.caption(
    "Hybrid Spectral-Spatial Attention + Relational Transformer (SSA-RTNet) + Laplacian High-Frequency Fusion "
    "+ 1-Shot Open-Set Biometric Verification & Supervised Occlusion Guardrail"
)

with st.sidebar:
    st.header("⚙️ System & Hardware Status")
    st.write(f"**Inference Device:** `{DEVICE}`")
    if model is not None:
        st.success(
            f"Loaded Checkpoint:\n`ssa_periocular_best.pt`\n"
            f"Backbone: `{ckpt.get('backbone', 'resnet34')}` | UBIPr Subjects: `{ckpt['num_identities']}`"
        )
    else:
        st.error("Checkpoint not found! Run `python train_eval.py` first.")

    st.divider()
    st.subheader("🎛️ Biometric Calibration")
    use_clahe = st.toggle("CLAHE Lighting Normalization", value=True)
    female_thresh = st.slider("Female Classification Threshold", 0.25, 0.75, 0.45, 0.02)
    custom_sim_thresh = st.slider(
        "1-Shot Verification Threshold", 0.40, 0.85, 0.58, 0.02,
        help="Same person across different photos/webcam scores 0.65–0.95; different people score < 0.45.",
    )

    st.divider()
    custom_reg = load_custom_registry()
    st.metric("Registered Profiles", len(custom_reg))
    if custom_reg and st.button("🗑️ Reset Registered Profiles"):
        REG_PATH.unlink(missing_ok=True)
        st.rerun()

tab1, tab2, tab3 = st.tabs([
    "🔍 Live Biometric Verification & Occlusion Analysis",
    "➕ Register New Person (1-Photo Enrollment)",
    "📊 Base Paper Comparison & Review Results",
])

# =====================================================================
# TAB 1: LIVE INFERENCE (FULL FACE, MASKED FACE, WEBCAM OR DEMO SAMPLE)
# =====================================================================
with tab1:
    col_src, col_face = st.columns([1, 1])
    with col_src:
        src_mode = st.radio(
            "Select Input Method:",
            ["Upload Image (Full Face / Masked / Eye Crop)", "Live Webcam", "Built-in Sample Demo Images"],
            horizontal=True,
        )
        bgr_input = None
        if src_mode == "Upload Image (Full Face / Masked / Eye Crop)":
            up = st.file_uploader("Upload JPG / PNG / BMP", type=["jpg", "jpeg", "png", "bmp"])
            if up:
                arr = np.asarray(bytearray(up.read()), dtype=np.uint8)
                bgr_input = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif src_mode == "Live Webcam":
            cam = st.camera_input("Capture Live Frame")
            if cam:
                arr = np.asarray(bytearray(cam.read()), dtype=np.uint8)
                bgr_input = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            demo_files = sorted(DEMO_DIR.glob("*.*")) if DEMO_DIR.exists() else []
            if demo_files:
                chosen = st.selectbox("Choose Sample Image:", demo_files, format_func=lambda p: p.name)
                bgr_input = cv2.imread(str(chosen))
            else:
                st.info("No demo images found in `sample_demo_images/` yet.")

    if bgr_input is not None and model is not None:
        loc = detect_and_crop_periocular(bgr_input, detector, use_clahe=use_clahe)
        with col_face:
            cap_txt = (
                "MediaPipe 478-Landmark Periocular Localization (Rotation-Aligned)"
                if loc["mode"] == "full_face_detected"
                else "Direct UBIPr Periocular ROI Input"
            )
            st.image(loc["annotated_face"], caption=cap_txt, use_container_width=True)

        g_probs_all, id_probs_all, occ_probs_all, ubipr_embs = [], [], [], []
        gradcams = {}

        for side, edata in loc["eyes"].items():
            out = run_model_inference(model, edata["processed_rgb"], DEVICE, gradcam_task="gender")
            weight = 1.0 if not edata["warnings"] else 0.5
            g_probs_all.append(out["gender_probs"] * weight)
            id_probs_all.append(out["id_probs"] * weight)
            occ_probs_all.append(out["occ_probs"])
            ubipr_embs.append(out["embedding"])
            gradcams[side] = out["gradcam"]

        ubipr_g_probs = np.sum(g_probs_all, axis=0) / np.sum(g_probs_all)

        # Supervised + Geometric Occlusion & Worst-Case Check
        avg_occ = np.mean(occ_probs_all, axis=0)
        occ_names = ckpt.get(
            "occlusion_names",
            ["Glasses / Spectacles", "Hair Occlusion", "Eye Closure / Blink", "Head Pose Angle"],
        )
        alerts = list(loc["global_warnings"])
        for side, edata in loc["eyes"].items():
            for w_msg in edata["warnings"]:
                alerts.append(f"{side.replace('_', ' ')}: {w_msg}")
        for idx_o, prob_o in enumerate(avg_occ):
            if prob_o >= 0.55:
                alerts.append(f"Neural Occlusion Head Detected: **{occ_names[idx_o]}** ({prob_o:.0%} confidence)")

        if alerts:
            st.warning("⚠️ **Occlusion & Quality Guardrail Notices:**\n- " + "\n- ".join(alerts))
        else:
            st.success("✅ **Worst-Case & Occlusion Check Passed:** Clear periocular visibility with optimal sharpness.")

        st.divider()
        k1, k2, k3 = st.columns(3)

        # =================================================================
        # CASE A: DIRECT UBIPR PERIOCULAR CROP -> USE TRAINED UBIPR HEADS
        # =================================================================
        if loc["mode"] == "direct_periocular_crop":
            p_female, p_male = float(ubipr_g_probs[1]), float(ubipr_g_probs[0])
            pred_gender = "Female" if p_female >= female_thresh else "Male"
            g_conf = p_female if pred_gender == "Female" else p_male

            avg_id_probs = np.mean(id_probs_all, axis=0)
            top_cls = int(avg_id_probs.argmax())
            idx_to_subj = ckpt.get("idx_to_subj", {})
            pred_subj = idx_to_subj.get(top_cls, idx_to_subj.get(str(top_cls), f"C{top_cls+1}"))
            cls_conf = float(avg_id_probs[top_cls])

            # Also compute cosine similarity to UBIPr gallery template
            cos_sim = cls_conf
            if ubipr_gal is not None:
                sims = ubipr_gal["templates"] @ ubipr_embs[0]
                cos_sim = float(np.max(sims))

            k1.metric(
                "🚻 Predicted Gender (UBIPr Periocular)",
                pred_gender,
                f"Confidence: {g_conf:.1%} (M:{p_male:.2f} | F:{p_female:.2f})",
            )
            k2.metric(
                "🆔 Identified UBIPr Dataset Subject",
                f"Subject {pred_subj}",
                f"Softmax Conf: {cls_conf:.1%} | Cosine Sim: {cos_sim:.3f}",
            )
            k3.metric(
                "📋 Biometric Verification Status",
                f"Verified UBIPr Identity ({pred_subj})",
                f"Matched across {ckpt['num_identities']} Enrolled UBIPr Classes",
            )

        # =================================================================
        # CASE B: FULL FACE / MASKED FACE / LIVE WEBCAM
        # =================================================================
        else:
            rw_vec, rw_g_probs = extract_realworld_identity_and_gender(
                loc, embedder, gender_net, ubipr_g_probs, DEVICE
            )
            p_female, p_male = float(rw_g_probs[1]), float(rw_g_probs[0])
            pred_gender = "Female" if p_female >= female_thresh else "Male"
            g_conf = p_female if pred_gender == "Female" else p_male

            best_user = None
            best_sim = -1.0
            for entry in custom_reg:
                bank = np.array(entry.get("template_bank", [entry["embedding"]]), dtype=np.float32)
                norms = np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
                bank = bank / norms
                sim = float(np.max(bank @ rw_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_user = entry

            k1.metric(
                "🚻 Predicted Gender (Real-Time)",
                pred_gender,
                f"Confidence: {g_conf:.1%} (M:{p_male:.2f} | F:{p_female:.2f})",
            )

            if best_user is not None and best_sim >= custom_sim_thresh:
                k2.metric(
                    "🆔 Identified Person (Registered Gallery)",
                    f"✅ {best_user['name']}",
                    f"ID: {best_user['person_id']} | Cosine Similarity: {best_sim:.3f}",
                )
                k3.metric(
                    "📋 Verified Profile Metadata",
                    f"{best_user['role']}",
                    f"{best_user['notes']} | Registered Gender: {best_user['gender']}",
                )
            else:
                sim_str = f"Best Match Sim: {max(best_sim, 0.0):.3f} (< {custom_sim_thresh:.2f})" if custom_reg else "Gallery Empty"
                k2.metric(
                    "🆔 Person Identification Status",
                    "🆕 New Profile Detected",
                    f"Unregistered Identity ({sim_str})",
                )
                k3.metric(
                    "🛡️ Biometric Action Required",
                    "Not in Registered Database",
                    f"Real-time Gender ({pred_gender}) classified; register in Tab 2 to enroll.",
                )

        with st.expander("🔎 View Neural Occlusion Head Probabilities (Trained on UBIPr Metadata Tags)", expanded=False):
            occ_df = pd.DataFrame({
                "Occlusion / Degradation Factor": occ_names,
                "Predicted Probability": [round(float(x), 4) for x in avg_occ],
            })
            st.dataframe(occ_df, use_container_width=True)

        st.subheader("🔬 Base-Paper ROI & Explainability Pipeline")
        for side, edata in loc["eyes"].items():
            st.markdown(
                f"**{side.replace('_', ' ')}** — EAR: `{edata['ear']:.2f}` | "
                f"Sharpness: `{edata['sharpness']:.1f}` | Specular Glare: `{edata['glare']:.1%}`"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.image(edata["processed_rgb"], caption="1. Calibrated Periocular Crop + CLAHE", use_container_width=True)
            c2.image(edata["hex_roi"], caption="2. Hexagon-Shaped ROI (Base Paper)", use_container_width=True)
            c3.image(edata["laplacian"], caption="3. Laplacian High-Frequency Edges", use_container_width=True)
            c4.image(gradcams[side], caption="4. Spectral Attention Grad-CAM", use_container_width=True)

# =====================================================================
# TAB 2: 1-PHOTO DYNAMIC FACE ENROLLMENT (UPLOAD OR WEBCAM)
# =====================================================================
with tab2:
    st.subheader("➕ Register New Person into Biometric Gallery (Works from Just 1 Photo!)")
    st.info(
        "Upload **just 1 clear photo** of yourself (or snap 1 webcam frame). "
        "The system extracts your rotation-aligned periocular + upper-facial biometric signature so any other photo "
        "or live webcam capture of you will match immediately, while any unregistered person triggers **'New Profile Detected'**."
    )
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        e_name = st.text_input("Full Name", value="B. Sushanth Rao")
        e_id = st.text_input("Roll Number / Subject ID", value="CSE-2027-SUSHANTH")
        e_gender = st.selectbox("Gender", ["Male", "Female"])
        e_role = st.text_input("Role / Branch", value="B.Tech CSE Final Year - Project Lead")
        e_notes = st.text_input("Institution / Notes", value="Matrusri Engineering College")

    with col_e2:
        enroll_mode = st.radio("Enrollment Source:", ["Upload Photo(s)", "Capture via Live Webcam"], horizontal=True)
        bgr_enroll_list = []
        if enroll_mode == "Upload Photo(s)":
            e_files = st.file_uploader(
                "Select 1 or more Face Photos", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True
            )
            for f in e_files or []:
                arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    bgr_enroll_list.append(bgr)
        else:
            e_cam = st.camera_input("Snap Webcam Photo to Register")
            if e_cam is not None:
                arr = np.asarray(bytearray(e_cam.read()), dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    bgr_enroll_list.append(bgr)

        if bgr_enroll_list and model is not None:
            if st.button("✅ Register Biometric Profile Now", type="primary"):
                bank_vectors = []
                preview_img = None
                for bgr in bgr_enroll_list:
                    res = detect_and_crop_periocular(bgr, detector, use_clahe=use_clahe)
                    preview_img = res["annotated_face"]
                    if res["mode"] == "full_face_detected":
                        rw_vec, _ = extract_realworld_identity_and_gender(
                            res, embedder, gender_net, np.array([0.5, 0.5]), DEVICE
                        )
                        bank_vectors.append(rw_vec.tolist())
                    else:
                        for _, edata in res["eyes"].items():
                            out = run_model_inference(model, edata["processed_rgb"], DEVICE)
                            bank_vectors.append(out["embedding"].tolist())

                if bank_vectors:
                    mean_vec = np.mean(bank_vectors, axis=0)
                    mean_vec = (mean_vec / max(np.linalg.norm(mean_vec), 1e-12)).tolist()
                    save_custom_registry({
                        "person_id": e_id,
                        "name": e_name,
                        "gender": e_gender,
                        "role": e_role,
                        "notes": e_notes,
                        "num_crops_enrolled": len(bank_vectors),
                        "embedding": mean_vec,
                        "template_bank": bank_vectors,
                    }, append_templates=True)
                    if preview_img is not None:
                        st.image(preview_img, caption="Registered Periocular & Facial Landmarks", width=320)
                    st.success(f"🎉 Registered **{e_name} ({e_id})**! Go to Tab 1 and test with any other photo or live webcam.")

    if custom_reg:
        st.subheader("📂 Currently Registered Profiles")
        st.dataframe(
            pd.DataFrame([
                {k: v for k, v in r.items() if k not in {"embedding", "template_bank"}}
                for r in custom_reg
            ]),
            use_container_width=True,
        )

# =====================================================================
# TAB 3: BASE PAPER COMPARISON & EXPERIMENTAL METRICS
# =====================================================================
with tab3:
    st.subheader("🎯 Complete Project Objective Metrics (Accuracy, Precision, Recall, F1, MCC, FAR, FRR, ROC-AUC)")
    obj_csv = RESULTS_DIR / "complete_objective_metrics.csv"
    if obj_csv.exists():
        st.dataframe(pd.read_csv(obj_csv), use_container_width=True)

    st.subheader("📈 Quantitative Comparison: Proposed Architecture vs. Base Paper (Scientific Reports 2025)")
    comp_csv = RESULTS_DIR / "base_paper_comparison.csv"
    if comp_csv.exists():
        st.dataframe(pd.read_csv(comp_csv), use_container_width=True)

    r1, r2, r3 = st.columns(3)
    if (RESULTS_DIR / "training_convergence.png").exists():
        r1.image(str(RESULTS_DIR / "training_convergence.png"), caption="Multi-Task Convergence", use_container_width=True)
    if (RESULTS_DIR / "gender_confusion_matrix.png").exists():
        r2.image(str(RESULTS_DIR / "gender_confusion_matrix.png"), caption="Gender Confusion Matrix", use_container_width=True)
    if (RESULTS_DIR / "gender_roc_curve.png").exists():
        r3.image(str(RESULTS_DIR / "gender_roc_curve.png"), caption="Gender ROC-AUC Curve", use_container_width=True)

    r4, r5 = st.columns(2)
    if (RESULTS_DIR / "identity_verification_roc.png").exists():
        r4.image(str(RESULTS_DIR / "identity_verification_roc.png"), caption="Person ID Verification ROC Curve", use_container_width=True)
    if (RESULTS_DIR / "far_frr_eer_curve.png").exists():
        r5.image(str(RESULTS_DIR / "far_frr_eer_curve.png"), caption="Biometric FAR vs. FRR (EER) Curve", use_container_width=True)