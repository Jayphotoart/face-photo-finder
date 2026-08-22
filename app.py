import streamlit as st
import cv2
import numpy as np
import json
import socket
import qrcode
import os
import shutil
import hashlib
import datetime
import tempfile
import faiss
import requests
import urllib.parse
import csv
import pandas as pd
import pickle
from insightface.app import FaceAnalysis
from face_search import find_best_global_assignment
from PIL import Image
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ============================================================
# SESSION STATE INIT
# ============================================================
if "pending_faces" not in st.session_state:
    st.session_state.pending_faces = []
if "cart" not in st.session_state:
    st.session_state.cart = []
if "payment_done" not in st.session_state:
    st.session_state.payment_done = False

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

st.set_page_config(page_title="જય ફોટો શોધ", page_icon="📸", layout="wide")

# ============================================================
# GOOGLE DRIVE INTEGRATION (જેમ છે તેમ રાખો)
# ============================================================
ROOT_FOLDER_ID = "1B-qd1ZtJkQfxIUzpUCxdvaVIMAkVQtqH"

def get_drive_service():
    """Google Drive Service - Service Account (Cloud) અથવા OAuth (લોકલ)"""
    
    # 1️⃣ Service Account (st.secrets માંથી) - આ Cloud માટે યોગ્ય છે
    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        return build('drive', 'v3', credentials=creds)
    except Exception:
        # Service Account ન મળે તો આગળ વધો
        pass

    # 2️⃣ શું આપણે Cloud પર છીએ? (Streamlit Cloud ને શોધો)
    is_cloud = os.environ.get('STREAMLIT_SHARING') or os.environ.get('STREAMLIT_CLOUD')
    if is_cloud:
        # Cloud પર OAuth શક્ય નથી, એટલે ભૂલ બતાવો
        st.error("❌ Google Drive Service Account સેટ નથી. કૃપા કરીને st.secrets માં 'gcp_service_account' ઉમેરો.")
        return None

    # 3️⃣ OAuth 2.0 (ફક્ત લોકલ માટે - token.pickle અને credentials.json ની જરૂર)
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        else:
            # credentials.json ફાઈલ હોવી જોઈએ, ન હોય તો error
            if not os.path.exists('credentials.json'):
                st.error("❌ 'credentials.json' ફાઈલ મળી નહીં. OAuth સેટ કરવા માટે તેને લોકલ ફોલ્ડરમાં મૂકો.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

def get_drive_folder_id(event_name):
    try:
        drive_service = get_drive_service()
        query = f"name = '{event_name}' and mimeType = 'application/vnd.google-apps.folder' and '{ROOT_FOLDER_ID}' in parents and trashed = false"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        items = results.get('files', [])
        if items:
            return items[0]['id']
        else:
            folder_metadata = {
                'name': event_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [ROOT_FOLDER_ID]
            }
            folder = drive_service.files().create(body=folder_metadata, fields='id', supportsAllDrives=True).execute()
            return folder.get('id')
    except Exception as e:
        st.error(f"❌ Google Drive API Error: {e}")
        return None

def upload_to_drive(file_path, folder_id):
    try:
        drive_service = get_drive_service()
        if drive_service is None:
            return None

        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)

        # 🔥 આ લાઇન ઉમેરો - Shared Drive માં અપલોડ કરવા માટે
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True   # <--- આ
        ).execute()
        return file.get('id')
    except Exception as e:
        st.error(f"Google Drive upload error: {e}")
        return None
    try:
        drive_service = get_drive_service()
        file_metadata = {'name': os.path.basename(file_path), 'parents': [folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id', supportsAllDrives=True).execute()
        return file.get('id')
    except Exception as e:
        st.error(f"Google Drive upload error: {e}")
        return None

# ============================================================
# LOCAL EVENT FUNCTIONS (ગુમ થયેલા ફંક્શન્સ)
# ============================================================
def get_event_dir(event_name):
    """લોકલ ફોલ્ડર પાથ બનાવો"""
    base = "events"
    event_path = os.path.join(base, event_name)
    photos_path = os.path.join(event_path, "images")
    os.makedirs(photos_path, exist_ok=True)
    return event_path, photos_path

def save_event_data_to_drive(event_name, data, folder_id):
    try:
        drive_service = get_drive_service()
        if drive_service is None:
            st.warning("⚠️ Google Drive ઉપલબ્ધ નથી.")
            return False

        # ... (અહીં temp_path બનાવવાનો કોડ છે) ...

        # Drive પર જૂની data.json શોધો
        query = f"name='data.json' and '{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query, 
            fields="files(id)",
            supportsAllDrives=True   # <--- આ ઉમેરો
        ).execute()
        for file in results.get('files', []):
            drive_service.files().delete(
                fileId=file['id'],
                supportsAllDrives=True   # <--- આ ઉમેરો
            ).execute()

        # નવી data.json અપલોડ કરો
        media = MediaFileUpload(temp_path, mimetype='application/json')
        file_metadata = {
            'name': 'data.json',
            'parents': [folder_id]
        }
        drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True   # <--- આ ઉમેરો
        ).execute()

        # ... (બાકીનો કોડ) ...
        return True
    except Exception as e:
        st.error(f"❌ Error saving to Drive: {e}")
        return False

def load_event_data_local(event_name):
    """લોકલ data.json વાંચો"""
    try:
        event_path, _ = get_event_dir(event_name)
        json_path = os.path.join(event_path, "data.json")
        if not os.path.exists(json_path):
            return {"password": "", "faces": []}
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            data = {"password": "", "faces": data}
        # એમ્બેડિંગ્સ યોગ્ય ફોર્મેટમાં લાવો
        for face in data.get("faces", []):
            if "embedding" in face and isinstance(face["embedding"], str):
                try:
                    face["embedding"] = json.loads(face["embedding"])
                except:
                    face["embedding"] = []
        return data
    except Exception:
        return {"password": "", "faces": []}

def list_all_local_events():
    """બધી લોકલ ઇવેન્ટ્સની યાદી"""
    base = "events"
    if not os.path.exists(base):
        return []
    events = []
    for item in os.listdir(base):
        path = os.path.join(base, item)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "data.json")):
            events.append(item)
    return events

# ============================================================
# HELPER: load/save event (લોકલ જ)
# ============================================================
def load_event_data(event_name):
    return load_event_data_local(event_name)

def save_event_data(event_name, data):
    return save_event_data_local(event_name, data)

# ============================================================
# ANALYTICS
# ============================================================
def log_activity(event_name, activity_type, person_label="", amount=0):
    log_file = "analytics.csv"
    file_exists = os.path.exists(log_file)
    try:
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Event", "Activity", "Person", "Amount"])
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                event_name,
                activity_type,
                person_label,
                amount
            ])
        return True
    except:
        return False

# ============================================================
# CSS, HEADER, SIDEBAR (એ જ રાખો, વધુ નહીં લખું)
# ============================================================
st.markdown("""<style> ... તમારું CSS અહીં મૂકો ... </style>""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    try:
        st.image("assets/logo.jpg", width=100)
    except:
        st.markdown("## 📸")
with col2:
    st.markdown("""
    <div class="brand-text" style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
        <h1 style="font-size: 2.8rem; font-weight: 900; color: #0f0f0f; margin: 0; letter-spacing: -1px;">
            JAY <span style="color: #d4af37;">PHOTO</span> SHODH
        </h1>
        <div style="font-size: 0.9rem; color: #6c757d; margin-top: -5px; font-weight: 400; letter-spacing: 2px;">
            ✨ AI POWERED PHOTO SEARCH
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.image("assets/logo.jpg", width="stretch")
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 1.5rem;">
    <div style="color: white; font-weight: 800; font-size: 1.4rem; margin: 0; letter-spacing: 1px;">
        JAY <span style="color: #d4af37;">PHOTO</span>
    </div>
    <div style="color: #adb5bd; font-size: 0.7rem; font-weight: 400; letter-spacing: 3px; margin-top: 2px;">
        ART
    </div>
</div>
""", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    "📌 પેજ પસંદ કરો",
    ["🔍 ફોટો શોધો", "📂 ઇવેન્ટ મેનેજ", "📱 QR કોડ બનાવો", "📊 Analytics", "📊 બેન્ચમાર્ક"],
    format_func=lambda x: x
)

# ============================================================
# PASSWORD PROTECTION (એડમિન)
# ============================================================
def check_password():
    if st.session_state.get("authenticated", False):
        return True
    st.sidebar.markdown("---")
    password = st.sidebar.text_input("🔒 એડમિન પાસવર્ડ:", type="password", key="admin_pass")
    if password:
        if password.strip() == "JayPhotoArt@2026":
            st.session_state.authenticated = True
            st.sidebar.success("✅ પ્રવેશ મળ્યો!")
            st.rerun()
            return True
        else:
            st.sidebar.error("❌ ખોટો પાસવર્ડ!")
            return False
    return False

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram_message(message):
    try:
        TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
        TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
    except:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except:
        return False

# ============================================================
# INSIGHTFACE & FAISS
# ============================================================
@st.cache_resource
def load_insightface():
    app = FaceAnalysis(name='buffalo_l', root='insightface_models')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

@st.cache_resource
def load_event_faiss_index(event_name):
    data = load_event_data(event_name)
    if not data or not data.get("faces"):
        return None, None
    valid_faces = []
    for item in data.get("faces", []):
        emb = parse_embedding(item.get("embedding"))
        if emb is not None:
            valid_faces.append(item)
    if not valid_faces:
        return None, None
    embeddings = np.array([item["embedding"] for item in valid_faces], dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, valid_faces

def parse_embedding(embedding_data):
    if embedding_data is None:
        return None
    if isinstance(embedding_data, str):
        try:
            return np.array(json.loads(embedding_data), dtype=np.float32)
        except:
            return None
    if isinstance(embedding_data, list):
        return np.array(embedding_data, dtype=np.float32)
    if isinstance(embedding_data, np.ndarray):
        return embedding_data
    return None

app = load_insightface()
PHOTO_PRICE = 10

# ============================================================
# PAGE 1: MANAGE EVENTS (સુધારેલ)
# ============================================================
if option == "📂 ઇવેન્ટ મેનેજ":
    st.markdown("""
    <div class="card">
        <div class="card-title">📂 ઇવેન્ટ મેનેજમેન્ટ</div>
        <div class="card-desc">અહીં તમે નવી ઇવેન્ટ બનાવી શકો છો, ફોટા અપલોડ કરી શકો છો અને ચહેરાઓને લેબલ આપી શકો છો.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("➕ નવી ઇવેન્ટ બનાવો", expanded=False):
        new_event = st.text_input("ઇવેન્ટનું નામ (દા.ત., શર્મા_લગ્ન)")
        event_password = st.text_input("🔒 ઇવેન્ટ પાસવર્ડ (ગ્રાહકો માટે)", type="password")
        if st.button("📌 ઇવેન્ટ બનાવો", key="create_event"):
            if new_event.strip() and event_password.strip():
                event_name = new_event.strip()
                initial_data = {"password": event_password, "faces": []}
                if save_event_data_local(event_name, initial_data):
                    # ફોલ્ડર પણ બને તે માટે get_event_dir કૉલ કરો
                    get_event_dir(event_name)
                    st.success(f"✅ ઇવેન્ટ '{event_name}' સફળતાપૂર્વક બની ગઈ!")
                    st.rerun()
                else:
                    st.error("❌ ઇવેન્ટ બનાવવામાં ભૂલ આવી.")
            else:
                st.error("❌ કૃપા કરીને નામ અને પાસવર્ડ બંને ભરો.")

    available_events = list_all_local_events()
    if not available_events:
        st.info("ℹ️ હજુ સુધી કોઈ ઇવેન્ટ નથી. ઉપર નવી ઇવેન્ટ બનાવો.")
    else:
        selected_event = st.selectbox("📁 ઇવેન્ટ પસંદ કરો", available_events)
        if selected_event:
            st.subheader(f"📸 ફોટા અપલોડ કરો - {selected_event}")
            uploaded_files = st.file_uploader(
                "ઇવેન્ટના ફોટા પસંદ કરો",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )

            if uploaded_files:
                if st.button("🚀 ફોટા પ્રોસેસ અને સેવ કરો"):
                    event_path, photos_path = get_event_dir(selected_event.strip())
                    if not os.path.exists(photos_path):
                        st.error("❌ ઇવેન્ટ ફોલ્ડર મળ્યું નહીં!")
                        st.stop()

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_files = len(uploaded_files)

                    event_data = load_event_data_local(selected_event.strip())
                    existing_faces = event_data.get("faces", [])
                    processed_count = 0
                    auto_saved_count = 0   # ✅ વ્યાખ્યા ઉમેરી

                    folder_id = get_drive_folder_id(selected_event.strip())  # Drive Folder ID
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"⏳ {file.name} પર કામ ચાલુ છે... ({i+1}/{total_files})")

                        # ૧. ફોટો લોકલ સેવ કરો
                        file_path = os.path.join(photos_path, file.name)
                        file.seek(0)
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                        # ---------- 🔥 નવો ફેરફાર: Drive પર અપલોડ ----------
                        drive_file_id = upload_to_drive(file_path, folder_id)
                        # જો Drive API ન ચાલે તો drive_file_id = None રહેશે, પણ કોઈ ભૂલ નહીં આવે
                        # -------------------------------------------------------

                        # ૨. ફેસ ડિટેક્શન
                        img = cv2.imread(file_path)
                        if img is None:
                            st.warning(f"⚠️ {file.name} વાંચવામાં ભૂલ આવી.")
                            continue

                        faces = app.get(img)
                        if len(faces) == 0:
                            st.warning(f"⚠️ {file.name} માં કોઈ ચહેરો મળ્યો નથી.")
                            continue

                        # ૩. દરેક ચહેરા માટે
                        for face_idx, face in enumerate(faces):
                            embedding_list = face.embedding.tolist()
                            # અત્યારે drive_file_id નથી, પછી upload_to_drive કરી શકો
                            drive_file_id = None  # અથવા upload_to_drive(file_path, folder_id) કરો
                            unique_name = file.name  # હાલ માટે ફાઈલ નામ

                            # ચહેરો ક્રોપ કરો
                            bbox = face.bbox.astype(int)
                            x1, y1, x2, y2 = bbox
                            pad = 20
                            h, w = img.shape[:2]
                            x1 = max(0, x1 - pad)
                            y1 = max(0, y1 - pad)
                            x2 = min(w, x2 + pad)
                            y2 = min(h, y2 + pad)
                            face_crop = img[y1:y2, x1:x2]
                            crop_filename = f"{hashlib.md5((unique_name + str(face_idx)).encode()).hexdigest()[:8]}.jpg"
                            crop_path = os.path.join("temp_crops", crop_filename)
                            os.makedirs("temp_crops", exist_ok=True)
                            cv2.imwrite(crop_path, face_crop)

                            embedding = face.embedding / np.linalg.norm(face.embedding)

                            # --- SMART AUTO-LABEL ---
                            matched_label = None
                            best_sim = 0.0
                            if existing_faces:
                                for item in existing_faces:
                                    db_emb = parse_embedding(item.get("embedding"))
                                    if db_emb is not None:
                                        sim = float(np.dot(embedding, db_emb))
                                        if sim > 0.65 and sim > best_sim:
                                            best_sim = sim
                                            matched_label = item.get("person_label")

                            if matched_label and matched_label != "SKIP":
                                existing_faces.append({
                                    "filename": unique_name,
                                    "drive_file_id": drive_file_id,
                                    "person_label": matched_label,
                                    "embedding": embedding.tolist()
                                })
                                auto_saved_count += 1
                                if os.path.exists(crop_path):
                                    os.remove(crop_path)
                            else:
                                st.session_state.pending_faces.append({
                                    "crop_path": crop_path,
                                    "embedding": embedding.tolist(),
                                    "original_filename": unique_name,
                                    "drive_file_id": drive_file_id,
                                    "label": "SKIP"
                                })

                        processed_count += 1
                        progress_bar.progress((i + 1) / total_files)

                    # ૪. ડેટા સેવ કરો
                    event_data["faces"] = existing_faces
                    save_event_data_local(selected_event.strip(), event_data)
                    save_event_data_to_drive(selected_event.strip(), event_data, folder_id)

                    status_text.empty()
                    st.success(f"✅ {processed_count} ફોટા સફળતાપૂર્વક પ્રોસેસ થયા! (ઓટો-સેવ: {auto_saved_count})")
                    st.rerun()

            # ---------- SMART GROUP LABELING ----------
            if st.session_state.pending_faces:
                st.subheader(f"🏷️ {len(st.session_state.pending_faces)} નવા ચહેરાઓને સ્માર્ટ ગ્રૂપમાં ગોઠવો")
                st.caption("🔍 સમાન દેખાતા નવા ચહેરાઓ એક ગ્રૂપમાં ગોઠવાયા છે. નામ આપો:")

                pending = st.session_state.pending_faces
                embeddings = np.array([face["embedding"] for face in pending], dtype=np.float32)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings_norm = embeddings / norms
                sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)

                threshold = 0.65
                n = len(pending)
                visited = [False] * n
                clusters = []
                for i in range(n):
                    if not visited[i]:
                        cluster = [i]
                        visited[i] = True
                        for j in range(i+1, n):
                            if not visited[j] and sim_matrix[i][j] > threshold:
                                cluster.append(j)
                                visited[j] = True
                        clusters.append(cluster)

                for group_idx, cluster in enumerate(clusters):
                    st.markdown(f"### 🎯 ગ્રૂપ {group_idx + 1} (કુલ {len(cluster)} ચહેરા)")
                    cols = st.columns(min(4, len(cluster)))
                    for col_idx, face_idx in enumerate(cluster):
                        col = cols[col_idx % 4]
                        with col:
                            face_data = pending[face_idx]
                            if os.path.exists(face_data["crop_path"]):
                                st.image(face_data["crop_path"], width=150)
                            else:
                                st.warning("⚠️ ફોટો મળ્યો નથી")

                    group_label = st.text_input(
                        f"ગ્રૂપ {group_idx + 1} ને નામ આપો",
                        value="",
                        key=f"group_label_{group_idx}",
                        placeholder="દા.ત., રાજેશ, પ્રિયા, A"
                    )
                    if group_label.strip():
                        for face_idx in cluster:
                            pending[face_idx]["label"] = group_label.strip()
                    else:
                        for face_idx in cluster:
                            pending[face_idx]["label"] = "SKIP"
                    st.divider()

                if st.button("💾 બધા લેબલ સેવ કરો", key="save_all_labels"):
                    event_data = load_event_data_local(selected_event.strip())
                    existing_faces = event_data.get("faces", [])
                    count = 0
                    for face_data in pending:
                        lbl = face_data["label"].strip()
                        if lbl != "SKIP" and lbl != "":
                            existing_faces.append({
                                "filename": face_data["original_filename"],
                                "drive_file_id": face_data["drive_file_id"],
                                "person_label": lbl,
                                "embedding": face_data["embedding"]
                            })
                            count += 1
                    event_data["faces"] = existing_faces
                    save_event_data_local(selected_event.strip(), event_data, folder_id)

                    # ક્રોપ ફાઈલો ડિલીટ કરો
                    for face_data in pending:
                        try:
                            if os.path.exists(face_data["crop_path"]):
                                os.remove(face_data["crop_path"])
                        except:
                            pass
                    st.session_state.pending_faces = []
                    st.cache_resource.clear()
                    st.success(f"✅ {count} નવા ચહેરા '{selected_event}' માં સેવ થયા!")
                    st.rerun()

            st.divider()
            event_data = load_event_data_local(selected_event.strip())
            faces_list = event_data.get("faces", [])
            st.write(f"📊 આ ઇવેન્ટમાં કુલ **{len(faces_list)}** લેબલ કરેલા ચહેરા છે.")
            if len(faces_list) > 0:
                st.subheader("🖼️ લેબલ કરેલા ફોટા")
                for i in range(0, len(faces_list), 4):
                    cols = st.columns(4)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(faces_list):
                            item = faces_list[idx]
                            file_id = item.get("drive_file_id")
                            if file_id:
                                img_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                                with col:
                                    st.image(img_url, caption=f"લેબલ: {item['person_label']}", width=150)
                            else:
                                # લોકલ ફોટો બતાવવા પ્રયાસ
                                local_img = os.path.join("events", selected_event, "images", item.get("filename", ""))
                                if os.path.exists(local_img):
                                    st.image(local_img, caption=f"લેબલ: {item['person_label']}", width=150)
                                else:
                                    st.write(f"❌ {item.get('filename', 'Unknown')} (Drive ID missing)")
            else:
                st.info("ℹ️ હજુ સુધી કોઈ ફોટો લેબલ થયો નથી.")

            # DELETE EVENT
            st.divider()
            st.markdown("### 🗑️ ઇવેન્ટ કાઢી નાખો")
            st.warning(f"⚠️ આ ઇવેન્ટ ('{selected_event}') અને તેના બધા લોકલ ફોટા કાયમ માટે ડિલીટ થઈ જશે!")
            if st.button(f"🗑️ '{selected_event}' ઇવેન્ટ કાઢી નાખો", type="primary"):
                try:
                    shutil.rmtree(os.path.join("events", selected_event))
                    st.success(f"✅ '{selected_event}' ઇવેન્ટ લોકલ પરથી ડિલીટ થઈ ગઈ!")
                    st.session_state.pending_faces = []
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ ઇવેન્ટ ડિલીટ કરતી વખતે ભૂલ: {e}")

# ============================================================
# PAGE 2: SEARCH FACE (જેમ છે તેમ, માત્ર load/save લોકલ વાપરો)
# ============================================================
elif option == "🔍 ફોટો શોધો":
    query_params = st.query_params
    event_name = query_params.get("event", None)

    if event_name is None:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔍 તમારા ફોટા શોધો</div>
            <div class="card-desc">કૃપા કરીને QR કોડ સ્કેન કરો અથવા ઇવેન્ટ લિંક ખોલો.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        event_folder = os.path.join("events", event_name)
        if not os.path.exists(event_folder):
            st.error(f"❌ '{event_name}' ઇવેન્ટ મળી નહીં. કૃપા કરીને યોગ્ય QR કોડ વાપરો.")
        else:
            if f"auth_{event_name}" not in st.session_state:
                st.session_state[f"auth_{event_name}"] = False

            if not st.session_state[f"auth_{event_name}"]:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">🔒 '{event_name}' ઇવેન્ટ માટે પાસવર્ડ</div>
                    <div class="card-desc">આ ઇવેન્ટને ઍક્સેસ કરવા માટે પાસવર્ડ લખો.</div>
                </div>
                """, unsafe_allow_html=True)
                entered_password = st.text_input("🔑 ઇવેન્ટ પાસવર્ડ:", type="password")
                if st.button("🚪 પ્રવેશ કરો"):
                    event_data = load_event_data_local(event_name)
                    if event_data.get("password") == entered_password:
                        st.session_state[f"auth_{event_name}"] = True
                        st.success("✅ પ્રવેશ મળ્યો!")
                        st.rerun()
                    else:
                        st.error("❌ ખોટો પાસવર્ડ!")
                st.stop()

            st.markdown(f"""
            <div class="card">
                <div class="card-title">🔍 '{event_name}' માં તમારા ફોટા શોધો</div>
                <div class="card-desc">નીચે તમારો ફોટો અપલોડ કરો અથવા સેલ્ફી લો, અમે તમારા બધા ફોટા શોધી આપીશું.</div>
            </div>
            """, unsafe_allow_html=True)

            index, db_data = load_event_faiss_index(event_name)
            if index is None or len(db_data) == 0:
                st.warning("ℹ️ આ ઇવેન્ટમાં હજુ સુધી કોઈ ફોટા નથી.")
            else:
                unique_labels = set()
                for item in db_data:
                    unique_labels.add(item["person_label"])
                persons_list = list(unique_labels)
                st.sidebar.success(f"✅ {len(db_data)} ચહેરા ઇન્ડેક્સ થયા")
                st.sidebar.info(f"👤 વ્યક્તિઓ: {', '.join(persons_list)}")

                st.subheader("📸 ફોટો અપલોડ કરવાની રીત")
                upload_option = st.radio(
                    "વિકલ્પ પસંદ કરો:",
                    ["📸 કેમેરાથી સેલ્ફી લો", "📁 ફોટો અપલોડ કરો"],
                    index=0,
                    key="upload_option"
                )
                uploaded_file = None
                if upload_option == "📸 કેમેરાથી સેલ્ફી લો":
                    uploaded_file = st.camera_input("📸 સેલ્ફી લો", key="camera_input")
                else:
                    uploaded_file = st.file_uploader("📁 ફોટો પસંદ કરો...", type=["jpg", "jpeg", "png"], key="file_uploader")

                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    img = cv2.imread(tmp_path)
                    if img is not None:
                        st.image(img, channels="BGR", caption="તમારો ફોટો", width=300)
                        with st.spinner("🔍 તમારા ફોટા શોધાઈ રહ્યા છે..."):
                            faces = app.get(img)
                            if len(faces) == 0:
                                st.warning("❌ ફોટામાં કોઈ ચહેરો દેખાયો નહીં!")
                            else:
                                st.success(f"✅ {len(faces)} ચહેરો શોધાયો!")
                                faces = sorted(faces, key=lambda f: f.bbox[0])

                                query_embeddings = []
                                for i, face in enumerate(faces):
                                    emb = face.embedding / np.linalg.norm(face.embedding)
                                    query_embeddings.append({"query_face": f"face_{i}", "embedding": emb.tolist()})

                                query_face_matches = {}
                                for q_data in query_embeddings:
                                    q_face = q_data["query_face"]
                                    q_emb = np.array(q_data["embedding"], dtype=np.float32).reshape(1, -1)
                                    k = min(10, len(db_data))
                                    scores, indices = index.search(q_emb, k)
                                    query_face_matches[q_face] = []
                                    for score, idx in zip(scores[0], indices[0]):
                                        if score > 0:
                                            query_face_matches[q_face].append({
                                                "person": db_data[idx]["person_label"],
                                                "similarity": float(score),
                                                "filename": db_data[idx]["filename"]
                                            })

                                result = find_best_global_assignment(
                                    query_embeddings,
                                    query_face_matches,
                                    persons_list
                                )

                                if result:
                                    for match in result:
                                        if match is None: continue
                                        q_face = match["query_face"]
                                        all_matches = query_face_matches.get(q_face, [])
                                        if len(all_matches) < 2:
                                            match["decision"] = "STRONG"
                                            continue
                                        top_score = all_matches[0]["similarity"]
                                        second_score = all_matches[1]["similarity"]
                                        margin = top_score - second_score
                                        if top_score > 0.80:
                                            match["decision"] = "STRONG"
                                        elif top_score > 0.65 and margin > 0.08:
                                            match["decision"] = "GOOD"
                                        elif margin < 0.05:
                                            match["decision"] = "AMBIGUOUS"
                                        else:
                                            match["decision"] = "WEAK"

                                st.subheader("📸 તમારા મેચ થયેલા ફોટા")
                                matched_persons = set()
                                for match in result:
                                    if match is not None and match['similarity'] > 0.30:
                                        matched_persons.add(match['person'])

                                if matched_persons:
                                    send_telegram_message(
                                        f"✅ <b>નવો ગ્રાહક!</b>\n"
                                        f"📸 ઇવેન્ટ: {event_name}\n"
                                        f"👤 વ્યક્તિ: {', '.join(matched_persons)}\n"
                                        f"🕒 {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}"
                                    )

                                    # CART SYSTEM (એ જ રાખો, અહીં ફક્ત થોડું સુધારેલ)
                                    for person in matched_persons:
                                        st.markdown(f"**👤 વ્યક્તિ: {person}**")
                                        person_photos = [item for item in db_data if item["person_label"] == person]
                                        if person_photos:
                                            for idx, item in enumerate(person_photos):
                                                if idx % 4 == 0:
                                                    cols = st.columns(4)
                                                col = cols[idx % 4]
                                                with col:
                                                    try:
                                                        file_id = item.get("drive_file_id")
                                                        if file_id:
                                                            img_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                                                            st.image(img_url, width=150)
                                                            img_path = img_url
                                                        else:
                                                            local_path = os.path.join("events", event_name, "images", item["filename"])
                                                            if os.path.exists(local_path):
                                                                st.image(local_path, width=150)
                                                                img_path = local_path
                                                            else:
                                                                st.write(f"📁 {item.get('filename', 'Unknown')}")
                                                                img_path = None
                                                    except Exception as e:
                                                        st.write(f"📁 {item.get('filename', 'Unknown')}")
                                                        img_path = None

                                                    price = PHOTO_PRICE
                                                    cart_key = f"cart_{person}_{idx}"
                                                    selected = st.checkbox(f"🛒 ₹{price}" if price > 0 else "🆓 FREE", key=cart_key)
                                                    if selected:
                                                        cart_item = {
                                                            "person": person,
                                                            "filename": item["filename"],
                                                            "price": price,
                                                            "img_path": img_path,
                                                            "drive_file_id": item.get("drive_file_id")
                                                        }
                                                        if cart_item not in st.session_state.cart:
                                                            st.session_state.cart.append(cart_item)
                                                    else:
                                                        st.session_state.cart = [c for c in st.session_state.cart if not (c["person"] == person and c["filename"] == item["filename"])]

                                            if st.button(f"➕ {person} ના બધા ફોટા કાર્ટમાં ઉમેરો", key=f"add_all_{person}"):
                                                for item in person_photos:
                                                    local_path = os.path.join("events", event_name, "images", item["filename"])
                                                    cart_item = {
                                                        "person": person,
                                                        "filename": item["filename"],
                                                        "price": PHOTO_PRICE,
                                                        "img_path": local_path if os.path.exists(local_path) else None,
                                                        "drive_file_id": item.get("drive_file_id")
                                                    }
                                                    if cart_item not in st.session_state.cart:
                                                        st.session_state.cart.append(cart_item)
                                                st.rerun()
                                        else:
                                            st.write("❌ આ વ્યક્તિના કોઈ ફોટા નથી.")

                                    # CART DISPLAY (સાઈડબાર)
                                    st.sidebar.markdown("---")
                                    st.sidebar.markdown("## 🛒 તમારું કાર્ટ")
                                    if st.session_state.cart:
                                        cart = st.session_state.cart
                                        total_price = sum(item["price"] for item in cart)
                                        for idx, item in enumerate(cart):
                                            if item['price'] == 0:
                                                st.sidebar.write(f"{idx+1}. {item['person']} - 🆓 FREE")
                                            else:
                                                st.sidebar.write(f"{idx+1}. {item['person']} - ₹{item['price']}")
                                        st.sidebar.markdown(f"### 💰 કુલ: ₹{total_price}")

                                        if st.sidebar.button("🗑️ કાર્ટ ખાલી કરો"):
                                            st.session_state.cart = []
                                            st.session_state.payment_done = False
                                            st.rerun()

                                        if st.sidebar.button(f"🧾 ચેકઆઉટ (₹{total_price})"):
                                            MY_UPI_ID = "dineshmakwna123@oksbi"
                                            MY_NAME = "Jay Photography"
                                            order_id = f"PHOTO_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                                            encoded_name = urllib.parse.quote(MY_NAME)
                                            encoded_note = urllib.parse.quote("Photo Download Payment")
                                            upi_links = {
                                                "📱 Google Pay": f"gpay://upi/pay?pa={MY_UPI_ID}&pn={encoded_name}&am={total_price}&cu=INR&tn={encoded_note}&tr={order_id}",
                                                "📱 PhonePe": f"phonepe://pay?pa={MY_UPI_ID}&pn={encoded_name}&am={total_price}",
                                                "📱 Paytm": f"paytmmp://pay?pa={MY_UPI_ID}&pn={encoded_name}&am={total_price}",
                                                "📱 BHIM": f"bhim://upi://pay?pa={MY_UPI_ID}&pn={encoded_name}&am={total_price}&cu=INR",
                                                "📱 Generic UPI": f"upi://pay?pa={MY_UPI_ID}&pn={encoded_name}&am={total_price}&cu=INR&tn={encoded_note}&tr={order_id}"
                                            }
                                            st.sidebar.markdown("### 💳 તમારી UPI એપ પસંદ કરો:")
                                            for label, link in upi_links.items():
                                                st.sidebar.link_button(label, link, width="stretch")
                                            st.sidebar.markdown("---")
                                            if st.sidebar.button("✅ પેમેન્ટ થઈ ગયું!", width="stretch"):
                                                unique_persons = set(item['person'] for item in cart)
                                                persons_text = ", ".join(unique_persons)
                                                send_telegram_message(
                                                    f"💰 <b>પેમેન્ટ મળ્યું!</b>\n"
                                                    f"📸 ઇવેન્ટ: {event_name}\n"
                                                    f"👤 ગ્રાહક(ઓ): {persons_text}\n"
                                                    f"💵 રકમ: ₹{total_price}\n"
                                                    f"🕒 {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}"
                                                )
                                                st.session_state.payment_done = True
                                                st.rerun()

                                        if st.session_state.get("payment_done", False):
                                            st.sidebar.markdown("---")
                                            st.sidebar.markdown("## 📥 તમારા ફોટા ડાઉનલોડ કરો")
                                            # ZIP ડાઉનલોડ (ફક્ત લોકલ ફોટા માટે)
                                            if st.sidebar.button("📥 બધા ફોટા ડાઉનલોડ કરો (ZIP)"):
                                                import zipfile, io
                                                zip_buffer = io.BytesIO()
                                                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                                                    for item in cart:
                                                        if item["img_path"] and os.path.exists(item["img_path"]):
                                                            zip_file.write(item["img_path"], item["filename"])
                                                zip_buffer.seek(0)
                                                st.sidebar.download_button(
                                                    label="📥 ZIP ડાઉનલોડ કરો",
                                                    data=zip_buffer,
                                                    file_name="my_photos.zip",
                                                    mime="application/zip",
                                                    key="zip_download_final"
                                                )
                                            for idx, item in enumerate(cart):
                                                if item["img_path"] and os.path.exists(item["img_path"]):
                                                    with open(item["img_path"], "rb") as f:
                                                        st.sidebar.download_button(
                                                            label=f"📸 {item['filename']} ડાઉનલોડ કરો",
                                                            data=f,
                                                            file_name=item["filename"],
                                                            mime="image/jpeg",
                                                            key=f"final_dl_{idx}"
                                                        )
                                            # શેરિંગ બટન્સ (એ જ)
                                            st.sidebar.markdown("---")
                                            st.sidebar.markdown("## 📤 તમારા ફોટા શેર કરો")
                                            app_url = "https://jayphotofinder.streamlit.app"
                                            share_text = "🌟 મારા ઇવેન્ટના સુંદર ફોટા જુઓ! જય ફોટો શોધ દ્વારા શોધ્યા."
                                            whatsapp_url = f"https://api.whatsapp.com/send?text={share_text} {app_url}"
                                            st.sidebar.markdown(f"[![WhatsApp](https://img.icons8.com/color/48/000000/whatsapp.png)]({whatsapp_url}) શેર કરો")
                                            facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={app_url}"
                                            st.sidebar.markdown(f"[![Facebook](https://img.icons8.com/color/48/000000/facebook.png)]({facebook_url}) શેર કરો")
                                            if st.sidebar.button("📋 લિંક કોપી કરો"):
                                                st.sidebar.code(app_url)
                                                st.sidebar.success("✅ લિંક કોપી થઈ ગઈ!")
                                            email_url = f"mailto:?subject=મારા ફોટા જુઓ&body={share_text} {app_url}"
                                            st.sidebar.markdown(f"[![Email](https://img.icons8.com/color/48/000000/gmail.png)]({email_url}) ઈમેઈલ દ્વારા શેર કરો")
                                            if st.sidebar.button("✅ ડાઉનલોડ થઈ ગયા! કાર્ટ ખાલી કરો"):
                                                st.session_state.cart = []
                                                st.session_state.payment_done = False
                                                st.rerun()
                                    else:
                                        st.sidebar.info("🛒 કાર્ટ ખાલી છે")
                                        st.session_state.payment_done = False

# ============================================================
# PAGE 3: QR CODE GENERATE (એ જ)
# ============================================================
elif option == "📱 QR કોડ બનાવો":
    st.markdown("""
    <div class="card">
        <div class="card-title">📱 QR કોડ બનાવો</div>
        <div class="card-desc">અહીં તમે કોઈ પણ ઇવેન્ટ માટે QR કોડ બનાવી શકો છો. ગ્રાહકો આ QR કોડ સ્કેન કરીને તેમના ફોટા શોધી શકશે.</div>
    </div>
    """, unsafe_allow_html=True)

    events = list_all_local_events()
    if not events:
        st.warning("⚠️ હજુ સુધી કોઈ ઇવેન્ટ નથી. કૃપા કરીને '📂 ઇવેન્ટ મેનેજ' માં પહેલાં ઇવેન્ટ બનાવો.")
    else:
        selected_event = st.selectbox("📂 ઇવેન્ટ પસંદ કરો", events)
        if selected_event:
            clean_event = selected_event.replace(" ", "_")
            url = f"https://jayphotofinder.streamlit.app/event={clean_event}"
            qr_img = qrcode.make(url)
            qr_img_array = np.array(qr_img.convert('RGB'))
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(qr_img_array, caption=f"📱 '{selected_event}' માટે QR કોડ", width=300)
                st.success(f"🔗 URL: {url}")
                st.caption("📌 ગ્રાહકો આ QR કોડ સ્કેન કરીને તેમના ફોટા જોઈ શકે છે.")
                from io import BytesIO
                buffered = BytesIO()
                qr_img.save(buffered, format="PNG")
                st.download_button(
                    label="⬇ QR કોડ ડાઉનલોડ કરો",
                    data=buffered.getvalue(),
                    file_name=f"qr_{clean_event}.png",
                    mime="image/png"
                )
            with col2:
                st.info("💡 કેવી રીતે વાપરવું?")
                st.write("1. આ QR કોડને પ્રિન્ટ કરીને ઇવેન્ટમાં મૂકો.")
                st.write("2. ગ્રાહકો ફોન વડે સ્કેન કરશે.")
                st.write("3. તેઓ સેલ્ફી લઈને તેમના ફોટા જોશે.")

# ============================================================
# PAGE 4: BENCHMARK (એ જ)
# ============================================================
else:
    st.header("📊 બેન્ચમાર્ક પરિણામો")
    try:
        df = pd.read_csv("benchmark_results.csv")
        st.dataframe(df)
        col1, col2, col3 = st.columns(3)
        top1_pass = (df['top1'] == "PASS").sum()
        exact_pass = (df['exact_ranking'] == "PASS").sum()
        avg_rank = df['ranking_accuracy'].mean()
        col1.metric("Top-1 Accuracy", f"{top1_pass}/9 ({top1_pass/9*100:.1f}%)")
        col2.metric("Exact Ranking", f"{exact_pass}/9 ({exact_pass/9*100:.1f}%)")
        col3.metric("Avg Rank Score", f"{avg_rank:.1f}%")
        st.bar_chart(df.set_index('test')['ranking_accuracy'])
    except FileNotFoundError:
        st.warning("benchmark_results.csv મળી નહીં.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    📸 <strong>જય ફોટો શોધ</strong> - AI દ્વારા તમારા ફોટા શોધો<br>
    © 2026 Jay Photography | Made with ❤️ in Gujarat
</div>
""", unsafe_allow_html=True)