import io
import hashlib
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="PELCON FARM AI", page_icon="🐑", layout="wide")
st.title("🐑 PELCON FARM AI")
st.caption("Δοκιμαστική ανίχνευση και καταμέτρηση προβάτων — χωρίς σύνδεση με κάμερες")

@st.cache_resource(show_spinner="Φόρτωση μοντέλου τεχνητής νοημοσύνης...")
def load_model():
    return YOLO("yolo11n.pt")

def detect(frame, confidence, tiled=True, nms_threshold=0.35):
    """Combine full-image and overlapping tile predictions, then deduplicate."""
    model = load_model()
    h, w = frame.shape[:2]
    regions = [(0, 0, w, h)]
    if tiled and min(h, w) >= 480:
        # Four overlapping tiles; overlap prevents splitting sheep at tile edges.
        tile_w, tile_h = int(w * 0.60), int(h * 0.60)
        for x in (0, w - tile_w):
            for y in (0, h - tile_h):
                regions.append((x, y, x + tile_w, y + tile_h))
    all_boxes, scores = [], []
    for x1, y1, x2, y2 in regions:
        crop = frame[y1:y2, x1:x2]
        result = model.predict(source=crop, conf=confidence, classes=[18],
                               verbose=False, imgsz=960)[0]
        if result.boxes is None:
            continue
        coords = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for b, score in zip(coords, confs):
            bx1, by1, bx2, by2 = b
            all_boxes.append([int(bx1 + x1), int(by1 + y1),
                              int(bx2 + x1), int(by2 + y1)])
            scores.append(float(score))
    if not all_boxes:
        return np.empty((0, 4), dtype=int)
    xywh = [[b[0], b[1], max(1, b[2]-b[0]), max(1, b[3]-b[1])] for b in all_boxes]
    keep = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=confidence, nms_threshold=nms_threshold)
    if len(keep) == 0:
        return np.empty((0, 4), dtype=int)
    return np.array([all_boxes[int(i)] for i in np.array(keep).flatten()], dtype=int)

def image_png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()

def correction_editor(rgb, boxes, identity, title="✏️ Διόρθωση"):
    st.subheader(title)
    state_key = "edited_" + identity
    nonce_key = "nonce_" + identity
    if state_key not in st.session_state:
        st.session_state[state_key] = [list(map(int, b)) for b in boxes]
    if nonce_key not in st.session_state:
        st.session_state[nonce_key] = 0
    edited = st.session_state[state_key]
    st.caption("Διάλεξε ενέργεια και πάτησε στην εικόνα. Η προσθήκη τοποθετεί νέο πλαίσιο στο σημείο. Η αφαίρεση διαγράφει μόνο πλαίσιο που πάτησες.")
    action = st.radio("Ενέργεια", ["Αφαίρεση", "Προσθήκη"], horizontal=True, key="action_" + identity)
    a, b = st.columns(2)
    width = a.slider("Πλάτος νέου πλαισίου", 30, 400, 110, 10, key="width_" + identity)
    height = b.slider("Ύψος νέου πλαισίου", 30, 400, 120, 10, key="height_" + identity)
    st.metric("Διορθωμένα πρόβατα", len(edited))
    display_width = min(1100, rgb.shape[1])
    scale = display_width / rgb.shape[1]
    preview = annotate(rgb, np.asarray(edited, dtype=int).reshape(-1, 4))
    preview = preview.resize((display_width, max(1, round(rgb.shape[0] * scale))))
    point = streamlit_image_coordinates(preview, key=f"editor_{identity}_{st.session_state[nonce_key]}")
    if point is not None:
        x, y = int(point["x"] / scale), int(point["y"] / scale)
        if action == "Προσθήκη":
            x1, y1 = max(0, x - width // 2), max(0, y - height // 2)
            edited.append([x1, y1, min(rgb.shape[1]-1, x1 + width), min(rgb.shape[0]-1, y1 + height)])
        else:
            inside = [i for i, (x1,y1,x2,y2) in enumerate(edited) if x1 <= x <= x2 and y1 <= y <= y2]
            if inside:
                # Prefer the smallest box under the click when boxes overlap.
                selected = min(inside, key=lambda i: (edited[i][2]-edited[i][0]) * (edited[i][3]-edited[i][1]))
                edited.pop(selected)
            else:
                st.info("Δεν υπάρχει πλαίσιο σε αυτό το σημείο. Πάτησε μέσα στο πράσινο πλαίσιο.")
        st.session_state[nonce_key] += 1
        st.rerun()
    if st.button("Επαναφορά πλαισίων", key="reset_" + identity):
        st.session_state[state_key] = [list(map(int, box)) for box in boxes]
        st.session_state[nonce_key] += 1
        st.rerun()
    output = annotate(rgb, np.asarray(edited, dtype=int).reshape(-1, 4))
    buf = io.BytesIO()
    output.save(buf, format="PNG")
    st.download_button("Λήψη διορθωμένης εικόνας", buf.getvalue(), file_name=f"pelcon_{identity}.png", mime="image/png", key="png_" + identity)
    df = pd.DataFrame([{"number": i, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]} for i, box in enumerate(edited, 1)])
    st.download_button("Λήψη διορθώσεων CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=f"pelcon_{identity}.csv", mime="text/csv", key="csv_" + identity)
    return edited

def annotate(rgb, boxes):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        draw.rectangle((int(x1), int(y1), int(x2), int(y2)), outline="#12C7A1", width=4)
        draw.text((int(x1) + 3, max(0, int(y1) - 17)), str(i), fill="#12C7A1", stroke_width=1, stroke_fill="black")
    return image

confidence = st.sidebar.slider("Ελάχιστη βεβαιότητα ανίχνευσης", 0.10, 0.90, 0.20, 0.05)
nms_threshold = st.sidebar.slider("Αφαίρεση επικαλυπτόμενων πλαισίων (NMS)", 0.20, 0.70, 0.35, 0.05, help="Μικρότερη τιμή αφαιρεί περισσότερα επικαλυπτόμενα πλαίσια, αλλά μπορεί να κρύψει γειτονικά πρόβατα.")
tiled = st.sidebar.checkbox("Ενισχυμένη ανίχνευση (τμήματα εικόνας)", value=True, help="Αναλύει την εικόνα σε επικαλυπτόμενα τμήματα. Πιο αργή αλλά συχνά βρίσκει μικρότερα ζώα.")
st.sidebar.info("Το γενικό μοντέλο ενδέχεται να παραλείψει πρόβατα που επικαλύπτονται.")
uploaded = st.file_uploader("Ανέβασε φωτογραφία ή σύντομο βίντεο", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        try:
            rgb = np.array(Image.open(uploaded).convert("RGB"))
            boxes = detect(rgb, confidence, tiled, nms_threshold)
            image_id = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
            corrected = correction_editor(rgb, boxes, image_id)
            actual = st.number_input("Πραγματικός αριθμός προβάτων", min_value=0, value=0, step=1)
            if actual:
                st.metric("Διαφορά", f"{len(corrected) - actual:+d}")
            h, w = rgb.shape[:2]
            labels = []
            for x1, y1, x2, y2 in corrected:
                labels.append(f"0 {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            archive_buffer = io.BytesIO()
            with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(f"images/{image_id}.png", image_png_bytes(rgb))
                archive.writestr(f"labels/{image_id}.txt", "\\n".join(labels) + "\\n")
                archive.writestr("data.yaml", "path: .\\ntrain: images\\nval: images\\nnames:\\n  0: sheep\\n")
            st.download_button("📦 Λήψη δεδομένων εκπαίδευσης (ZIP)", archive_buffer.getvalue(), file_name=f"pelcon_labels_{image_id}.zip", mime="application/zip")
        except Exception as exc:
            st.error(f"Δεν ήταν δυνατή η ανάλυση εικόνας: {exc}")
    else:
        st.warning("Στα βίντεο υπολογίζεται ο αριθμός προβάτων ανά καρέ, όχι ο συνολικός αριθμός μοναδικών ζώων.")
        if uploaded.size > 100 * 1024 * 1024:
            st.error("Για την πρώτη δοκιμή ανέβασε βίντεο έως 100 MB.")
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if not cap.isOpened():
                    raise ValueError("Το βίντεο δεν άνοιξε. Δοκίμασε MP4/H.264.")
                stride = max(1, round(fps * 2))
                max_samples = 30
                records = []
                sample_frames = []
                preview = st.empty()
                progress = st.progress(0)
                frame_idx = 0
                while len(records) < max_samples:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % stride == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        boxes = detect(rgb, confidence, tiled, nms_threshold)
                        records.append({"second": round(frame_idx / fps, 1), "detected_sheep": len(boxes)})
                        sample_frames.append((rgb.copy(), boxes.copy()))
                        preview.image(annotate(rgb, boxes), caption=f"Καρέ στα {frame_idx / fps:.1f} δευτ.", use_container_width=True)
                        progress.progress(min(1.0, len(records) / max_samples))
                    frame_idx += 1
                cap.release()
                if records:
                    df = pd.DataFrame(records)
                    st.metric("Μέγιστες ανιχνεύσεις σε ένα καρέ", int(df.detected_sheep.max()))
                    st.line_chart(df.set_index("second")["detected_sheep"])
                    st.dataframe(df, use_container_width=True)
                    st.subheader("Επίλεξε καρέ για διόρθωση")
                    selected_frame = st.selectbox("Χρόνος καρέ", list(range(len(records))), format_func=lambda i: f"{records[i]['second']} δευτ. — {records[i]['detected_sheep']} ανιχνεύσεις")
                    frame_rgb, frame_boxes = sample_frames[selected_frame]
                    video_id = hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]
                    correction_editor(frame_rgb, frame_boxes, f"{video_id}_frame_{selected_frame}", "✏️ Διόρθωση επιλεγμένου καρέ βίντεο")
                    st.download_button("Λήψη CSV βίντεο", df.to_csv(index=False).encode("utf-8-sig"), file_name="pelcon_video_counts.csv", mime="text/csv")
                else:
                    st.error("Δεν βρέθηκαν καρέ για ανάλυση.")
            except Exception as exc:
                st.error(f"Δεν ήταν δυνατή η ανάλυση βίντεο: {exc}")
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

st.divider()
st.caption("Πιλοτική εφαρμογή. Δεν παρέχει ακόμη αναγνώριση συγκεκριμένου ζώου ή αντιστοίχιση RFID.")
