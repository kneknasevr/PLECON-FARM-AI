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
            session_key = "corrections_" + image_id
            if session_key not in st.session_state:
                st.session_state[session_key] = [list(map(int, box)) for box in boxes]
            corrected = st.session_state[session_key]

            st.subheader("✏️ Χειροκίνητη διόρθωση")
            st.caption("Πάτησε πάνω σε πλαίσιο για αφαίρεση ή πάνω σε πρόβατο που λείπει για προσθήκη. Οι διορθώσεις μένουν προσωρινά στη συνεδρία· κατέβασε το ZIP για να τις κρατήσεις.")
            mode = st.radio("Ενέργεια στο πάτημα της εικόνας", ["Αφαίρεση πλαισίου", "Προσθήκη πλαισίου"], horizontal=True)
            col_w, col_h = st.columns(2)
            with col_w:
                new_width = st.slider("Πλάτος νέου πλαισίου (pixels)", 30, 400, 110, 10)
            with col_h:
                new_height = st.slider("Ύψος νέου πλαισίου (pixels)", 30, 400, 120, 10)

            st.metric("Διορθωμένος αριθμός προβάτων", len(corrected))
            display_width = min(1100, rgb.shape[1])
            scale = display_width / rgb.shape[1]
            display_height = max(1, round(rgb.shape[0] * scale))
            preview = annotate(rgb, np.asarray(corrected, dtype=int).reshape(-1, 4))
            preview = preview.resize((display_width, display_height))
            click_key = "click_nonce_" + image_id
            if click_key not in st.session_state:
                st.session_state[click_key] = 0
            click = streamlit_image_coordinates(preview, key=f"annotate_{image_id}_{st.session_state[click_key]}")
            if click is not None:
                x, y = int(click["x"] / scale), int(click["y"] / scale)
                if mode == "Προσθήκη πλαισίου":
                    x1 = max(0, x - new_width // 2)
                    y1 = max(0, y - new_height // 2)
                    x2 = min(rgb.shape[1] - 1, x1 + new_width)
                    y2 = min(rgb.shape[0] - 1, y1 + new_height)
                    corrected.append([x1, y1, x2, y2])
                elif corrected:
                    containing = [i for i, (x1,y1,x2,y2) in enumerate(corrected) if x1 <= x <= x2 and y1 <= y <= y2]
                    candidates = containing if containing else list(range(len(corrected)))
                    nearest = min(candidates, key=lambda i: ((corrected[i][0]+corrected[i][2])/2-x)**2 + ((corrected[i][1]+corrected[i][3])/2-y)**2)
                    corrected.pop(nearest)
                st.session_state[click_key] += 1
                st.rerun()

            if st.button("Επαναφορά αρχικών ανιχνεύσεων"):
                st.session_state[session_key] = [list(map(int, box)) for box in boxes]
                st.session_state[click_key] += 1
                st.rerun()

            actual = st.number_input("Πραγματικός αριθμός προβάτων (εφόσον τον γνωρίζεις)", min_value=0, value=0, step=1)
            if actual > 0:
                difference = len(corrected) - actual
                st.metric("Διαφορά διορθωμένης καταμέτρησης", f"{difference:+d}")
                st.caption("Η συμφωνία αριθμού δεν αποδεικνύει ότι όλα τα πλαίσια είναι σωστά.")

            final_image = annotate(rgb, np.asarray(corrected, dtype=int).reshape(-1, 4))
            image_buffer = io.BytesIO()
            final_image.save(image_buffer, format="PNG")
            st.download_button("Λήψη διορθωμένης εικόνας", image_buffer.getvalue(), file_name="pelcon_corrected.png", mime="image/png")
            csv = pd.DataFrame([{"number": i, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for i, b in enumerate(corrected, 1)])
            st.download_button("Λήψη διορθωμένων CSV", csv.to_csv(index=False).encode("utf-8-sig"), file_name="pelcon_corrected.csv", mime="text/csv")

            # YOLO dataset format: class x_center y_center width height (normalized).
            h, w = rgb.shape[:2]
            labels = []
            for x1, y1, x2, y2 in corrected:
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
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
