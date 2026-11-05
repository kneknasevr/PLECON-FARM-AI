"""Permanent sheep registry backed by Supabase Auth, Postgres and private Storage."""
import io
import uuid
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from supabase import create_client

def _visual_features(raw):
    """Experimental appearance comparison, NOT an individual animal ID model."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.thumbnail((512, 512))
    rgb = np.asarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (9, 8))
    dhash = (gray[:, 1:] > gray[:, :-1]).flatten()
    return hist, dhash

def _appearance_score(first, second):
    color = max(0.0, cv2.compareHist(first[0], second[0], cv2.HISTCMP_CORREL))
    shape = 1.0 - float(np.mean(first[1] != second[1]))
    return round(100 * (0.65 * color + 0.35 * shape), 1)

def render_registry():
    st.header("🐑 Μητρώο προβάτων — μόνιμη αποθήκευση")
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_PUBLISHABLE_KEY"]
    except (KeyError, FileNotFoundError):
        st.info("Η βάση είναι έτοιμη. Για να ενεργοποιηθεί το μητρώο, πρόσθεσε SUPABASE_URL και SUPABASE_PUBLISHABLE_KEY στα Streamlit Secrets.")
        return
    client = create_client(url, key)
    if "pelcon_auth" not in st.session_state:
        st.session_state.pelcon_auth = None
    auth = st.session_state.pelcon_auth
    if not auth:
        with st.form("pelcon_login"):
            email = st.text_input("Email")
            password = st.text_input("Κωδικός", type="password")
            submitted = st.form_submit_button("Σύνδεση στο μητρώο")
        if submitted:
            try:
                response = client.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.pelcon_auth = {"access_token": response.session.access_token, "refresh_token": response.session.refresh_token}
                st.rerun()
            except Exception:
                st.error("Αποτυχία σύνδεσης. Έλεγξε τα στοιχεία σου ή δημιούργησε χρήστη από το Supabase Authentication.")
        return
    try:
        client.auth.set_session(auth["access_token"], auth["refresh_token"])
        session = client.auth.get_session()
        user = client.auth.get_user().user
        if not user:
            raise ValueError("No authenticated user")
        st.session_state.pelcon_auth = {"access_token": session.access_token, "refresh_token": session.refresh_token}
    except Exception:
        st.session_state.pelcon_auth = None
        st.warning("Η σύνδεση έληξε. Συνδέσου ξανά.")
        return
    if st.button("Αποσύνδεση μητρώου"):
        client.auth.sign_out()
        st.session_state.pelcon_auth = None
        st.rerun()
    with st.form("new_sheep", clear_on_submit=True):
        tag = st.text_input("Αριθμός ενωτίου *")
        name = st.text_input("Όνομα (προαιρετικό)")
        notes = st.text_area("Σημειώσεις")
        if st.form_submit_button("Αποθήκευση προβάτου"):
            if not tag.strip():
                st.error("Συμπλήρωσε αριθμό ενωτίου.")
            else:
                try:
                    client.table("sheep").insert({"ear_tag": tag.strip(), "name": name.strip() or None, "notes": notes.strip() or None}).execute()
                    st.success("Το πρόβατο αποθηκεύτηκε μόνιμα.")
                except Exception as exc:
                    st.error(f"Αποτυχία αποθήκευσης (πιθανόν διπλός αριθμός): {exc}")
    try:
        rows = client.table("sheep").select("id,ear_tag,name,notes,created_at").order("created_at", desc=True).execute().data
    except Exception as exc:
        st.error(f"Αποτυχία φόρτωσης μητρώου: {exc}")
        return
    st.caption(f"Καταχωρισμένα ζώα: {len(rows)}")
    if rows:
        st.dataframe([{"Ενώτιο": r["ear_tag"], "Όνομα": r.get("name"), "Σημειώσεις": r.get("notes")} for r in rows], use_container_width=True)
        labels = {r["id"]: r["ear_tag"] for r in rows}
        selected = st.selectbox("Επίλεξε πρόβατο για φωτογραφίες", list(labels), format_func=lambda k: labels[k])
        current = next(row for row in rows if row["id"] == selected)
        with st.expander("✏️ Αλλαγή αριθμού προβάτου", expanded=False):
            with st.form("edit_sheep_number"):
                new_tag = st.text_input("Νέος αριθμός / ενώτιο", value=current["ear_tag"])
                new_name = st.text_input("Όνομα (προαιρετικό)", value=current.get("name") or "")
                if st.form_submit_button("💾 Αποθήκευση νέου αριθμού"):
                    new_tag = new_tag.strip()
                    if not new_tag:
                        st.error("Ο αριθμός δεν μπορεί να είναι κενός.")
                    elif any(row["ear_tag"] == new_tag and row["id"] != selected for row in rows):
                        st.error("Αυτός ο αριθμός χρησιμοποιείται ήδη από άλλο πρόβατο.")
                    else:
                        try:
                            client.table("sheep").update({"ear_tag": new_tag, "name": new_name.strip() or None}).eq("id", selected).execute()
                            st.success("Ο αριθμός ενημερώθηκε. Οι αποθηκευμένες φωτογραφίες παραμένουν στο ίδιο ζώο.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Αποτυχία ενημέρωσης: {exc}")
        photos = st.file_uploader("Πρόσθεσε φωτογραφίες αναφοράς", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="sheep_photos_upload")
        if st.button("Μόνιμη αποθήκευση φωτογραφιών", disabled=not photos):
            for photo in photos:
                try:
                    raw = photo.getvalue()
                    if len(raw) > 10 * 1024 * 1024:
                        st.error(f"{photo.name}: υπερβαίνει τα 10 MB")
                        continue
                    img = Image.open(io.BytesIO(raw))
                    img.verify()
                    ext = "png" if (photo.type == "image/png") else "webp" if photo.type == "image/webp" else "jpg"
                    mime = {"png":"image/png","webp":"image/webp","jpg":"image/jpeg"}[ext]
                    path = f"{user.id}/{selected}/{uuid.uuid4().hex}.{ext}"
                    client.storage.from_("sheep-photos").upload(path, raw, {"content-type": mime})
                    try:
                        client.table("sheep_photos").insert({"sheep_id": selected, "storage_path": path}).execute()
                    except Exception:
                        client.storage.from_("sheep-photos").remove([path])
                        raise
                    st.success(f"Αποθηκεύτηκε: {photo.name}")
                except Exception as exc:
                    st.error(f"{photo.name}: {exc}")
        saved = client.table("sheep_photos").select("id,storage_path,created_at").eq("sheep_id",selected).execute().data
        st.caption(f"Μόνιμα αποθηκευμένες φωτογραφίες για το ζώο: {len(saved)}")
        for item in saved:
            try:
                signed = client.storage.from_("sheep-photos").create_signed_url(item["storage_path"], 120)
                st.image(signed["signedURL"], width=200)
            except Exception:
                st.caption("Η προεπισκόπηση φωτογραφίας δεν είναι διαθέσιμη.")

        st.divider()
        st.subheader("🔍 Δοκιμή οπτικής σύγκρισης")
        st.caption("Συγκρίνει χρώμα και γενική εικόνα με τις φωτογραφίες αναφοράς του επιλεγμένου ζώου. Δεν αποτελεί αξιόπιστη ατομική ταυτοποίηση.")
        candidate = st.file_uploader("Νέα φωτογραφία για δοκιμή", type=["jpg", "jpeg", "png", "webp"], key="identity_test_upload")
        if st.button("Σύγκριση με φωτογραφίες αναφοράς", disabled=not candidate or not saved):
            try:
                probe = _visual_features(candidate.getvalue())
                scores = []
                for reference in saved:
                    try:
                        raw = client.storage.from_("sheep-photos").download(reference["storage_path"])
                        scores.append(_appearance_score(probe, _visual_features(raw)))
                    except Exception:
                        continue
                if not scores:
                    st.error("Δεν ήταν δυνατή η ανάγνωση φωτογραφιών αναφοράς από το Supabase.")
                else:
                    st.image(candidate, caption="Νέα φωτογραφία", width=280)
                    st.metric("Μεγαλύτερη οπτική ομοιότητα", f"{max(scores):.1f}%")
                    st.write(f"Συγκρίθηκαν {len(scores)} αποθηκευμένες φωτογραφίες του ζώου {labels[selected]}.")
                    st.warning("Το ποσοστό είναι δείκτης οπτικής ομοιότητας, ΟΧΙ πιθανότητα ότι πρόκειται για το ίδιο πρόβατο. Δοκίμασε και φωτογραφία διαφορετικού ζώου.")
            except Exception as exc:
                st.error(f"Αποτυχία σύγκρισης: {exc}")

def render_video_candidates(candidates):
    """Review and explicitly save video track crops to the existing sheep registry."""
    if not candidates:
        return
    st.subheader("💾 Αποθήκευση εντοπισμένων ζώων")
    st.warning("Τα Track IDs είναι προσωρινά: το ίδιο πρόβατο μπορεί να εμφανίζεται πολλές φορές. Επίλεξε μόνο ζώα που έχεις ελέγξει.")
    keys = sorted(candidates, key=lambda x: int(x))
    st.caption(f"Βρέθηκαν {len(keys)} φωτογραφίες από προσωρινά Track IDs. Οι φωτογραφίες είναι προεπισκοπήσεις, όχι βεβαιωμένες ταυτότητες.")
    for key in keys[:30]:
        with st.container():
            st.image(candidates[key], caption=f"Πρόβατο / Track {key}", width=210)
    if len(keys) > 30:
        st.info("Εμφανίζονται οι πρώτες 30 φωτογραφίες. Μπορείς να επιλέξεις και τις υπόλοιπες από τη λίστα.")
    chosen = st.multiselect("Επίλεξε τα ζώα που θέλεις να αποθηκεύσεις", keys,
                            format_func=lambda x: f"Track {x}")
    auth = st.session_state.get("pelcon_auth")
    if not auth:
        st.warning("Για να αποθηκεύσεις τις επιλεγμένες φωτογραφίες, συνδέσου πρώτα στο Μητρώο προβάτων στο πάνω μέρος της σελίδας.")
        return
    try:
        client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_PUBLISHABLE_KEY"])
        client.auth.set_session(auth["access_token"], auth["refresh_token"])
        user = client.auth.get_user().user
        if not user:
            st.error("Η σύνδεση στο μητρώο έληξε.")
            return
    except Exception as exc:
        st.error(f"Δεν είναι διαθέσιμο το μητρώο: {exc}")
        return
    if st.button("💾 Αποθήκευση επιλεγμένων με φωτογραφίες", disabled=not chosen):
        saved_count = 0
        for key in chosen:
            raw = candidates[key]
            # Unique provisional label. The sheep row ID remains stable after renaming.
            tag = f"AUTO-{uuid.uuid4().hex[:10].upper()}"
            sheep_id = None
            path = None
            try:
                row = client.table("sheep").insert({
                    "ear_tag": tag,
                    "notes": f"Προσωρινή καταχώριση από βίντεο, Track ID {key}. Απαιτείται επιβεβαίωση ταυτότητας."
                }).execute().data
                sheep_id = row[0]["id"]
                path = f"{user.id}/{sheep_id}/{uuid.uuid4().hex}.jpg"
                client.storage.from_("sheep-photos").upload(path, raw, {"content-type": "image/jpeg"})
                client.table("sheep_photos").insert({"sheep_id": sheep_id, "storage_path": path}).execute()
                st.success(f"Track {key} → {tag} (με φωτογραφία)")
                saved_count += 1
            except Exception as exc:
                if path:
                    try:
                        client.storage.from_("sheep-photos").remove([path])
                    except Exception:
                        pass
                if sheep_id:
                    try:
                        client.table("sheep").delete().eq("id", sheep_id).execute()
                    except Exception:
                        pass
                st.error(f"Track {key}: {exc}")
        if saved_count:
            st.info("Άνοιξε το Μητρώο προβάτων για να διορθώσεις τους προσωρινούς αριθμούς. Η επόμενη ανάλυση δεν αναγνωρίζει ακόμη αυτόματα τα ίδια ζώα.")
