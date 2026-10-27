"""Permanent sheep registry backed by Supabase Auth, Postgres and private Storage."""
import io
import uuid
import streamlit as st
from PIL import Image
from supabase import create_client

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
