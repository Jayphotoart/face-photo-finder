import streamlit as st
import cv2
import numpy as np
import json
import qrcode
import os
import shutil
import tempfile
import faiss
import pandas as pd
from io import BytesIO
from insightface.app import FaceAnalysis

# === GOOGLE DRIVE IMPORTS ===
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# Session state initialization
if "pending_faces" not in st.session_state:
    st.session_state.pending_faces = []

# ============================================================
# PAGE CONFIG
# ============================================================
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
st.set_page_config(page_title="જય ફોટો શોધ", page_icon="📸", layout="wide")

# ============================================================
# GOOGLE DRIVE FUNCTIONS
# ============================================================
SCOPES = ['https://www.googleapis.com/auth/drive']

@st.cache_resource
def get_drive_service():
    try:
        # Streamlit Secrets માંથી Google Drive ના credentials લેશે
        creds_info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error("❌ Google Drive Credentials મળ્યા નથી. કૃપા કરીને secrets.toml ચેક કરો.")
        return None

def get_or_create_drive_folder(service, folder_name, parent_id=None):
    """ડ્રાઇવમાં ફોલ્ડર શોધશે, ન હોય તો નવું બનાવશે."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    
    if not items:
        file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        if parent_id:
            file_metadata['parents'] = [parent_id]
        folder = service.files().create(body=file_metadata, fields='id').execute()
        
        # પબ્લિક એક્સેસ આપો જેથી એપમાં ફોટા દેખાય
        service.permissions().create(fileId=folder.get('id'), body={'type': 'anyone', 'role': 'reader'}).execute()
        return folder.get('id')
    return items[0]['id']

def upload_to_drive(service, file_path, file_name, folder_id, mime_type='image/jpeg'):
    """ફાઈલને ડ્રાઈવમાં અપલોડ કરશે."""
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=False)  # અહી False કર્યું છે
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# ============================================================
# LOCAL STORAGE (ટૂંકા ગાળા માટે અને JSON માટે)
# ============================================================
BASE_STORAGE_DIR = "events_data"
os.makedirs(BASE_STORAGE_DIR, exist_ok=True)

def get_event_dir(event_name):
    clean_name = event_name.strip().replace(" ", "_")
    event_path = os.path.join(BASE_STORAGE_DIR, clean_name)
    os.makedirs(event_path, exist_ok=True)
    return event_path

def save_event_data_local(event_name, data):
    event_path = get_event_dir(event_name)
    data_file = os.path.join(event_path, f"{event_name}_data.json")
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_event_data_local(event_name):
    event_path = get_event_dir(event_name)
    data_file = os.path.join(event_path, f"{event_name}_data.json")
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"password": "", "faces": [], "drive_folder_id": ""}

def list_all_local_events():
    if not os.path.exists(BASE_STORAGE_DIR):
        return []
    return [d for d in os.listdir(BASE_STORAGE_DIR) if os.path.isdir(os.path.join(BASE_STORAGE_DIR, d))]

# ============================================================
# INSIGHTFACE MODEL
# ============================================================
@st.cache_resource
def load_insightface():
    app = FaceAnalysis(name='buffalo_l', root='insightface_models')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def parse_embedding(embedding_data):
    if embedding_data is None: return None
    if isinstance(embedding_data, list): return np.array(embedding_data, dtype=np.float32)
    return None

app = load_insightface()
PHOTO_PRICE = 10

# ============================================================
# HEADER & SIDEBAR
# ============================================================
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("assets/logo.jpg"): st.image("assets/logo.jpg", width=100)
    else: st.markdown("## 📸")
with col2:
    st.markdown("""
    <div style="margin-top: 10px;">
        <h1 style="font-size: 2.2rem; font-weight: 900; margin:0;">
            JAY <span style="color: #d4af37;">PHOTO</span> SHODH
        </h1>
        <div style="color: #6c757d; font-size: 0.85rem; letter-spacing: 2px;">
            ✨ AI POWERED PHOTO SEARCH
        </div>
    </div>
    """, unsafe_allow_html=True)

option = st.sidebar.selectbox("📌 પેજ પસંદ કરો", ["🔍 ફોટો શોધો", "📂 ઇવેન્ટ મેનેજ", "📱 QR કોડ બનાવો"])

# ============================================================
# PAGE 1: MANAGE EVENTS & GOOGLE DRIVE UPLOAD
# ============================================================
if option == "📂 ઇવેન્ટ મેનેજ":
    st.subheader("📂 ઇવેન્ટ મેનેજમેન્ટ (Google Drive)")
    service = get_drive_service()
    
    with st.expander("➕ નવી ઇવેન્ટ બનાવો", expanded=False):
        new_event = st.text_input("ઇવેન્ટનું નામ (દા.ત., sharma_wedding)")
        event_password = st.text_input("🔒 ઇવેન્ટ પાસવર્ડ (ગ્રાહકો માટે)", type="password")
        
        if st.button("📌 ઇવેન્ટ બનાવો"):
            if new_event.strip() and event_password.strip() and service:
                clean_name = new_event.strip().replace(" ", "_")
                
                # ૧. ડ્રાઇવમાં મેઈન ફોલ્ડર બનાવો
                root_id = get_or_create_drive_folder(service, "JayPhotoShodh_Events")
                # ૨. તેની અંદર ઇવેન્ટનું ફોલ્ડર બનાવો
                event_folder_id = get_or_create_drive_folder(service, clean_name, root_id)
                
                initial_data = {"password": event_password.strip(), "faces": [], "drive_folder_id": event_folder_id}
                save_event_data_local(clean_name, initial_data)
                
                st.success(f"✅ ઇવેન્ટ '{clean_name}' સફળતાપૂર્વક Drive પર બની ગઈ!")
                st.rerun()
            else:
                st.error("❌ નામ, પાસવર્ડ ભરો અને ખાતરી કરો કે Drive કનેક્ટ છે.")

    available_events = list_all_local_events()

    if not available_events:
        st.info("ℹ️ હજુ સુધી કોઈ ઇવેન્ટ નથી.")
    else:
        selected_event = st.selectbox("📁 ઇવેન્ટ પસંદ કરો", available_events)
        
        if selected_event and service:
            st.markdown(f"### 📸 ફોટા અપલોડ કરો - `{selected_event}`")
            uploaded_files = st.file_uploader("ઇવેન્ટના ફોટા પસંદ કરો", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
            
            if uploaded_files and st.button("🚀 ફોટા Drive માં સેવ કરો"):
                event_data = load_event_data_local(selected_event)
                existing_faces = event_data.get("faces", [])
                drive_folder_id = event_data.get("drive_folder_id")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_files = len(uploaded_files)
                new_pending = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"⏳ {file.name} Drive માં અપલોડ થઈ રહ્યો છે... ({i+1}/{total_files})")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    # ૧. Drive માં ફોટો અપલોડ કરો
                    file_id = upload_to_drive(service, tmp_path, file.name, drive_folder_id)
                    
                    # ૨. ચહેરો સ્કેન કરો
                    img = cv2.imread(tmp_path)
                    if img is not None:
                        faces = app.get(img)
                        for j, face in enumerate(faces):
                            norm_emb = face.embedding / np.linalg.norm(face.embedding)
                            
                            # ચહેરો ઓળખવાનો પ્રયાસ (Auto-match)
                            matched_label = None
                            for ef in existing_faces:
                                db_emb = parse_embedding(ef.get("embedding"))
                                if db_emb is not None and float(np.dot(norm_emb, db_emb)) > 0.65:
                                    matched_label = ef.get("person_label")
                                    break
                            
                            if matched_label and matched_label != "SKIP":
                                existing_faces.append({
                                    "photo_name": file.name,
                                    "drive_file_id": file_id,
                                    "person_label": matched_label,
                                    "embedding": norm_emb.tolist()
                                })
                            else:
                                new_pending.append({
                                    "photo_name": file.name,
                                    "drive_file_id": file_id,
                                    "embedding": norm_emb.tolist(),
                                    "label": "Guest"
                                })
                    
                    os.remove(tmp_path)
                    progress_bar.progress((i + 1) / total_files)
                
                event_data["faces"] = existing_faces
                save_event_data_local(selected_event, event_data)
                
                # અજાણ્યા ચહેરાઓને સીધા ગેસ્ટ (Guest) તરીકે સેવ કરી દઈએ (ઝડપ માટે)
                if new_pending:
                    for pf in new_pending:
                        existing_faces.append({
                            "photo_name": pf["photo_name"],
                            "drive_file_id": pf["drive_file_id"],
                            "person_label": pf["label"],
                            "embedding": pf["embedding"]
                        })
                    event_data["faces"] = existing_faces
                    save_event_data_local(selected_event, event_data)
                
                st.cache_resource.clear()
                status_text.empty()
                st.success("✅ બધા ફોટા Google Drive પર અપલોડ થઈ ગયા અને ચહેરા સ્કેન થઈ ગયા!")
                st.rerun()

            st.divider()
            event_data = load_event_data_local(selected_event)
            faces_list = event_data.get("faces", [])
            st.write(f"📊 આ ઇવેન્ટમાં કુલ **{len(faces_list)}** ઓળખાયેલા ચહેરાઓ છે (Drive માં સેવ્ડ).")
            
            if st.button(f"🗑️ '{selected_event}' ઇવેન્ટ ડિલીટ કરો", type="primary"):
                event_path = get_event_dir(selected_event)
                if os.path.exists(event_path): shutil.rmtree(event_path)
                st.cache_resource.clear()
                st.success("✅ ઇવેન્ટ લોકલમાંથી ડિલીટ થઈ ગઈ!")
                st.rerun()

# ============================================================
# PAGE 2: SEARCH FACE
# ============================================================
elif option == "🔍 ફોટો શોધો":
    query_params = st.query_params
    event_name = query_params.get("event", None)
    
    available_events = list_all_local_events()
    if not available_events:
        st.info("ℹ️ કોઈ ઇવેન્ટ ઉપલબ્ધ નથી.")
        st.stop()
        
    if event_name not in available_events:
        event_name = st.selectbox("📂 ઇવેન્ટ પસંદ કરો", available_events)
    
    event_data = load_event_data_local(event_name)
    
    if event_data.get("password"):
        if f"auth_{event_name}" not in st.session_state:
            st.session_state[f"auth_{event_name}"] = False
            
        if not st.session_state[f"auth_{event_name}"]:
            entered_pw = st.text_input("🔑 ઇવેન્ટ પાસવર્ડ લખો:", type="password")
            if st.button("🚪 પ્રવેશ કરો"):
                if entered_pw == event_data.get("password"):
                    st.session_state[f"auth_{event_name}"] = True
                    st.rerun()
                else:
                    st.error("❌ ખોટો પાસવર્ડ!")
            st.stop()
            
    st.markdown(f"### 🔍 `{event_name}` માં તમારા ફોટા શોધો")
    upload_opt = st.radio("રીત પસંદ કરો:", ["📸 સેલ્ફી લો", "📁 ફોટો અપલોડ કરો"], horizontal=True)
    up_file = st.camera_input("📸 સેલ્ફી લો") if upload_opt == "📸 સેલ્ફી લો" else st.file_uploader("📁 ફોટો પસંદ કરો", type=["jpg", "jpeg", "png"])
    
    if up_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(up_file.read())
            tmp_p = tmp.name
            
        q_img = cv2.imread(tmp_p)
        if q_img is not None:
            faces = app.get(q_img)
            if not faces:
                st.warning("❌ ફોટામાં કોઈ ચહેરો મળ્યો નહીં!")
            else:
                q_emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
                matched = []
                seen_photos = set()
                
                for f_item in event_data.get("faces", []):
                    db_emb = parse_embedding(f_item.get("embedding"))
                    if db_emb is not None:
                        sim = float(np.dot(q_emb, db_emb))
                        if sim >= 0.45 and f_item["photo_name"] not in seen_photos:
                            matched.append((f_item["drive_file_id"], f_item["photo_name"]))
                            seen_photos.add(f_item["photo_name"])
                                
                if matched:
                    st.success(f"🎉 તમારા {len(matched)} ફોટા મળ્યા! (Drive માંથી)")
                    cols = st.columns(3)
                    for idx, (f_id, p_name) in enumerate(matched):
                        with cols[idx % 3]:
                            # સીધી ગૂગલ ડ્રાઇવની લિંકથી ફોટો બતાવશે
                            drive_url = f"https://drive.google.com/uc?export=view&id={f_id}"
                            st.image(drive_url, caption=p_name, use_container_width=True)
                            st.link_button("⬇️ ડાઉનલોડ", drive_url)
                else:
                    st.warning("🔍 આ ઇવેન્ટમાં તમારો કોઈ ફોટો મળ્યો નથી.")

# ============================================================
# PAGE 3: GENERATE QR CODE
# ============================================================
elif option == "📱 QR કોડ બનાવો":
    st.subheader("📱 QR કોડ બનાવો")
    events = list_all_local_events()
    if not events:
        st.warning("⚠️ પહેલાં ઇવેન્ટ બનાવો.")
    else:
        selected_event = st.selectbox("📂 ઇવેન્ટ પસંદ કરો", events)
        if selected_event:
            # ✅ અહી તમારું નવું ડોમેન સેવ થયેલું છે!
            url = f"https://jayphotoart.in/?event={selected_event}"
            qr = qrcode.make(url)
            buf = BytesIO()
            qr.save(buf, format="PNG")
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(buf.getvalue(), width=260, caption=f"QR for {selected_event}")
                st.download_button("⬇️ QR ડાઉનલોડ કરો", buf.getvalue(), file_name=f"{selected_event}_qr.png", mime="image/png")
            with c2:
                st.info("💡 ગ્રાહક આ QR સ્કેન કરશે એટલે સીધું આ પેજ ખુલશે:")
                st.code(url)

st.markdown("""
<div style="text-align: center; padding: 2rem 0 0.5rem 0; border-top: 1px solid #f0f0f0; color: #adb5bd; font-size: 0.8rem; margin-top:3rem;">
    📸 <strong>જય ફોટો શોધ</strong> - AI દ્વારા તમારા ફોટા શોધો<br>
    © 2026 Jay Photography | Made with ❤️ in Gujarat
</div>
""", unsafe_allow_html=True)