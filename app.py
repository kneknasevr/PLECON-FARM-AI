import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="PELCON FARM AI", page_icon="🐑", layout="wide")
st.title("🐑 PELCON FARM AI")
st.caption("Δοκιμαστική ανίχνευση και καταμέτρηση προβάτων — χωρίς σύνδεση με κάμερες")

@st.cache_resource(show_spinner="Φόρτωση μοντέλου τεχνητής νοημοσύνης...")
def load_model():
    return YOLO("yolo11n.pt")

def detect(frame, confidence, tiled=True):
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
    keep = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=confidence, nms_threshold=0.40)
    if len(keep) == 0:
        return np.empty((0, 4), dtype=int)
    return np.array([all_boxes[int(i)] for i in np.array(keep).flatten()], dtype=int)

def annotate(rgb, boxes):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        draw.rectangle((int(x1), int(y1), int(x2), int(y2)), outline="#12C7A1", width=4)
        draw.text((int(x1) + 3, max(0, int(y1) - 17)), str(i), fill="#12C7A1", stroke_width=1, stroke_fill="black")
    return image

confidence = st.sidebar.slider("Ελάχιστη βεβαιότητα ανίχνευσης", 0.10, 0.90, 0.20, 0.05)
tiled = st.sidebar.checkbox("Ενισχυμένη ανίχνευση (τμήματα εικόνας)", value=True, help="Αναλύει την εικόνα σε επικαλυπτόμενα τμήματα. Πιο αργή αλλά συχνά βρίσκει μικρότερα ζώα.")
st.sidebar.info("Το γενικό μοντέλο ενδέχεται να παραλείψει πρόβατα που επικαλύπτονται.")
uploaded = st.file_uploader("Ανέβασε φωτογραφία ή σύντομο βίντεο", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        try:
            rgb = np.array(Image.open(uploaded).convert("RGB"))
            boxes = detect(rgb, confidence, tiled)
            output = annotate(rgb, boxes)
            st.metric("Ανιχνεύσεις προβάτων", len(boxes))
            st.caption("Δοκιμαστική καταμέτρηση: ελέγξτε οπτικά για παραλείψεις και διπλά πλαίσια.")
            st.image(output, caption="Αριθμημένα πλαίσια ανίχνευσης", use_container_width=True)
            buf = io.BytesIO()
            output.save(buf, format="PNG")
            st.download_button("Λήψη εικόνας με πλαίσια", buf.getvalue(), file_name="pelcon_count.png", mime="image/png")
            csv = pd.DataFrame([{"number": i, "x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3])} for i, b in enumerate(boxes, 1)])
            st.download_button("Λήψη αποτελεσμάτων CSV", csv.to_csv(index=False).encode("utf-8-sig"), file_name="pelcon_count.csv", mime="text/csv")
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
                preview = st.empty()
                progress = st.progress(0)
                frame_idx = 0
                while len(records) < max_samples:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % stride == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        boxes = detect(rgb, confidence, tiled)
                        records.append({"second": round(frame_idx / fps, 1), "detected_sheep": len(boxes)})
                        preview.image(annotate(rgb, boxes), caption=f"Καρέ στα {frame_idx / fps:.1f} δευτ.", use_container_width=True)
                        progress.progress(min(1.0, len(records) / max_samples))
                    frame_idx += 1
                cap.release()
                if records:
                    df = pd.DataFrame(records)
                    st.metric("Μέγιστες ανιχνεύσεις σε ένα καρέ", int(df.detected_sheep.max()))
                    st.line_chart(df.set_index("second")["detected_sheep"])
                    st.dataframe(df, use_container_width=True)
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
