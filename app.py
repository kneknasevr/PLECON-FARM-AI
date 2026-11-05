import io
import hashlib
import zipfile
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="PELCON FARM AI", page_icon="🐑", layout="wide")
st.title("🐑 PELCON FARM AI")
from sheep_registry import render_registry
with st.expander("📒 Μητρώο προβάτων (μόνιμη αποθήκευση)", expanded=False):
    render_registry()
st.caption("Δοκιμαστική ανίχνευση και καταμέτρηση προβάτων — χωρίς σύνδεση με κάμερες")
st.info("Το μπλε περίγραμμα είναι σχηματικό, όχι πραγματική ανάλυση στάσης ή βάδισης.")

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
    """Render counted sheep with an illustrative animal silhouette or bounding boxes.

    The silhouette is a visualization derived from detection rectangles, NOT
    anatomical pose estimation. Do not use its legs to assess lameness.
    """
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    style = st.session_state.get("overlay_style", "Σχηματικό περίγραμμα")
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        if style == "Ορθογώνια πλαίσια":
            draw.rectangle((x1, y1, x2, y2), outline="#12C7A1", width=3)
        else:
            # Approximate sheep shape within the detection box. This is not
            # measured limb positioning, gait tracking, or pose inference.
            p = lambda fx, fy: (int(x1 + fx * w), int(y1 + fy * h))
            spine = [p(.12,.24), p(.26,.13), p(.58,.12), p(.76,.22)]
            head = [p(.76,.22), p(.85,.32), p(.94,.42), p(.91,.55), p(.82,.54)]
            belly = [p(.82,.54), p(.67,.68), p(.27,.67), p(.12,.56), p(.12,.24)]
            legs = [
                [p(.25,.66),p(.23,.85),p(.20,.97)],
                [p(.37,.67),p(.40,.86),p(.39,.97)],
                [p(.65,.67),p(.64,.86),p(.61,.97)],
                [p(.76,.62),p(.78,.84),p(.80,.97)],
            ]
            for line in [spine, head, belly, *legs]:
                draw.line(line, fill="#345BFF", width=max(2, min(5, w // 60)), joint="curve")
            draw.ellipse((x1 + int(.87*w), y1 + int(.39*h), x1 + int(.89*w), y1 + int(.41*h)), fill="#345BFF")
        draw.text((x1 + 3, max(0, y1 - 18)), str(i), fill="#12C7A1", stroke_width=2, stroke_fill="black")
    return image

st.sidebar.radio("Εμφάνιση καταμετρημένων προβάτων", ["Σχηματικό περίγραμμα", "Ορθογώνια πλαίσια"], key="overlay_style", help="Το περίγραμμα είναι ενδεικτικό και δεν αναγνωρίζει πραγματικές αρθρώσεις ή χωλότητα.")

_live_camera_lock = threading.Lock()

with st.expander("📱 Live κάμερα κινητού — καταμέτρηση", expanded=False):
    st.caption("Πάτησε START, επίτρεψε την κάμερα και στόχευσε τα πρόβατα. Μετρά τα ζώα στο καρέ, όχι μοναδικά ζώα που περνούν.")
    st.warning("Πιλοτική λειτουργία WebRTC. Σε ορισμένα δίκτυα 5G μπορεί να μην συνδέεται χωρίς TURN server.")
    st.link_button("🌐 Άνοιγμα Live σε Chrome / εξωτερικό browser", "https://pelcon-farm-ai.streamlit.app/", help="Άνοιξε τον σύνδεσμο σε Chrome, όχι μέσα στο APK. Μετά άνοιξε ξανά αυτή την ενότητα.")
    st.caption("Αν το START δεν εμφανίζεται: άνοιξε από Chrome, ανανέωσε τη σελίδα και επίτρεψε την κάμερα όταν ζητηθεί.")
    live_state = {"count": 0}

    def live_video_frame(frame):
        bgr = frame.to_ndarray(format="bgr24")
        h, w = bgr.shape[:2]
        scale = min(1.0, 640.0 / max(h, w))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)))
        if _live_camera_lock.acquire(blocking=False):
            try:
                prediction = load_model().predict(bgr, classes=[18], conf=0.3, imgsz=640, verbose=False)[0]
                boxes = prediction.boxes.xyxy.cpu().numpy().astype(int) if prediction.boxes is not None else []
                live_state["count"] = len(boxes)
                for n, (x1, y1, x2, y2) in enumerate(boxes, 1):
                    cv2.rectangle(bgr, (x1,y1), (x2,y2), (255,90,20), 2)
                    cv2.putText(bgr, str(n), (x1,max(18,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,90,20), 2)
            except Exception:
                cv2.putText(bgr, "AI error", (12,65), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
            finally:
                _live_camera_lock.release()
        cv2.rectangle(bgr, (4,4), (255,42), (0,0,0), -1)
        cv2.putText(bgr, f"Sheep in frame: {live_state['count']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    # The phone sends video to the AI server; it does not need a return camera stream.
    live_context = webrtc_streamer(
        key="pelcon-live-phone-camera",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video":{"facingMode":"environment"},"audio":False},
        video_frame_callback=live_video_frame,
        async_processing=True,
        desired_playing_state=False,
    )
    if live_context.state.playing:
        st.success("Η ζωντανή μετάδοση είναι ενεργή. Ο μετρητής εμφανίζεται πάνω στο βίντεο.")
    else:
        st.info("Αναμονή σύνδεσης. Αν δεν εμφανίζεται START, δοκίμασε από Chrome. Αν εμφανίζεται αλλά δεν συνδέεται μέσω 5G, μπορεί να απαιτείται TURN.")

    st.divider()
    st.markdown("**📸 Εναλλακτική: άμεση φωτογραφία και καταμέτρηση**")
    st.caption("Αν δεν εμφανίζεται το START, πάτησε το κουμπί παρακάτω. Η φωτογραφία αναλύεται αμέσως, χωρίς ξεχωριστό ανέβασμα.")
    phone_photo = st.camera_input("Άνοιγμα κάμερας / Λήψη φωτογραφίας", key="phone_camera_snapshot")
    if phone_photo is not None:
        try:
            phone_rgb = np.array(Image.open(phone_photo).convert("RGB"))
            phone_boxes = detect(phone_rgb, confidence=0.30, tiled=True, nms_threshold=0.35)
            st.metric("Πρόβατα στη φωτογραφία", len(phone_boxes))
            st.image(annotate(phone_rgb, phone_boxes), caption="Αριθμημένα ζώα (ενδεικτικά περιγράμματα)", use_container_width=True)
        except Exception as error:
            st.error(f"Η καταμέτρηση απέτυχε: {error}")

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
        st.info("Νέα λειτουργία: παρακολούθηση προβάτων με προσωρινά IDs (ByteTrack). Ο αριθμός είναι εκτίμηση μοναδικών διαδρομών, όχι εγγυημένα διαφορετικών ζώων.")
        max_mb = 500
        if uploaded.size > max_mb * 1024 * 1024:
            st.error(f"Το βίντεο υπερβαίνει τα {max_mb} MB.")
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    raise ValueError("Το βίντεο δεν άνοιξε. Δοκίμασε MP4/H.264.")
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Keep temporal gaps short enough for the tracker to maintain IDs.
                stride = max(1, round(fps / 5))
                max_processed = 900
                st.caption("Ανάλυση έως 900 δειγματοληπτικών καρέ (περίπου 3 λεπτά στα 5 fps).")
                records, sample_frames, unique_ids = [], [], set()
                preview = st.empty()
                progress = st.progress(0)
                frame_idx, processed = 0, 0
                model = load_model()
                while processed < max_processed:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % stride == 0:
                        # persist=True keeps tracker identities across processed frames.
                        result = model.track(source=frame, persist=True, tracker="bytetrack.yaml",
                                             conf=confidence, classes=[18], verbose=False, imgsz=960)[0]
                        ids = []
                        if result.boxes is not None and result.boxes.id is not None:
                            ids = [int(x) for x in result.boxes.id.cpu().numpy()]
                        unique_ids.update(ids)
                        count = len(result.boxes) if result.boxes is not None else 0
                        records.append({"second": round(frame_idx / fps, 1), "detected_sheep": count,
                                        "tracked_in_frame": len(ids), "cumulative_track_ids": len(unique_ids)})
                        if processed % 20 == 0:
                            shown = result.plot()
                            preview.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
                                          caption=f"Χρόνος {frame_idx/fps:.1f} δευτ. — μοναδικά track IDs: {len(unique_ids)}",
                                          use_container_width=True)
                        processed += 1
                        progress.progress(min(1.0, processed / max_processed))
                    frame_idx += 1
                cap.release()
                if records:
                    df = pd.DataFrame(records)
                    st.metric("Εκτίμηση διαφορετικών προβάτων (track IDs)", len(unique_ids))
                    st.metric("Μέγιστα πρόβατα ταυτόχρονα σε καρέ", int(df.detected_sheep.max()))
                    st.caption("Προσοχή: αν ένα ζώο χαθεί και επανεμφανιστεί, μπορεί να λάβει νέο ID και να διπλομετρηθεί. Σε πυκνό κοπάδι η εκτίμηση μπορεί να είναι ανακριβής.")
                    if processed >= max_processed:
                        st.warning("Έγινε ανάλυση μόνο του πρώτου τμήματος του βίντεο λόγω ορίου επεξεργασίας.")
                    st.line_chart(df.set_index("second")[["detected_sheep", "cumulative_track_ids"]])
                    st.dataframe(df, use_container_width=True)
                    st.download_button("Λήψη CSV παρακολούθησης",
                                       df.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="pelcon_video_tracking.csv", mime="text/csv")
                else:
                    st.error("Δεν βρέθηκαν καρέ για ανάλυση.")

            except Exception as exc:
                st.error(f"Δεν ήταν δυνατή η ανάλυση βίντεο: {exc}")
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

st.divider()
st.caption("Πιλοτική εφαρμογή. Δεν παρέχει ακόμη αναγνώριση συγκεκριμένου ζώου ή αντιστοίχιση RFID.")
