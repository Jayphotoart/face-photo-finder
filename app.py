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
import webbrowser
import requests
import urllib.parse
import csv
import pandas as pd
import pickle                         # ✅ NEW
from insightface.app import FaceAnalysis
from face_search import find_best_global_assignment
from PIL import Image
from google.oauth2 import service_account   # (તમે આને રાખી શકો છો અથવા કાઢી શકો છો)
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow   # ✅ NEW
from google.auth.transport.requests import Request       # ✅ NEW

# Session state initialization
if "pending_faces" not in st.session_state:
    st.session_state.pending_faces = []

# ============================================================
# ENVIRONMENT VARIABLE (OpenCV માટે)
# ============================================================
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="જય ફોટો શોધ",
    page_icon="📸",
    layout="wide"
)

# ============================================================
# LOCAL STORAGE SYSTEM (Streamlit Server Storage)
# ============================================================
import os
import json

BASE_STORAGE_DIR = "events_data"
os.makedirs(BASE_STORAGE_DIR, exist_ok=True)

def get_event_dir(event_name):
    """ઇવેન્ટ માટે લોકલ ફોલ્ડર બનાવે છે"""
    event_path = os.path.join(BASE_STORAGE_DIR, event_name)
    photos_path = os.path.join(event_path, "photos")
    os.makedirs(photos_path, exist_ok=True)
    return event_path, photos_path

def save_event_data_local(event_name, data):
    """ઇવેન્ટનો ડેટા અને ફેસ એમ્બેડિંગ્સ JSON માં સેવ કરે છે"""
    try:
        event_path, _ = get_event_dir(event_name)
        data_file = os.path.join(event_path, f"{event_name}_data.json")
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"❌ ડેટા સેવ કરતી વખતે ભૂલ: {e}")
        return False

def load_event_data_local(event_name):
    """ઇવેન્ટનો ડેટા લોડ કરે છે"""
    try:
        event_path, _ = get_event_dir(event_name)
        data_file = os.path.join(event_path, f"{event_name}_data.json")
        if os.path.exists(data_file):
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"password": "", "faces": []}
    except Exception as e:
        st.error(f"❌ ડેટા લોડ કરતી વખતે ભૂલ: {e}")
        return {"password": "", "faces": []}

def list_all_local_events():
    """બધી ઉપલબ્ધ ઇવેન્ટ્સનું લિસ્ટ મેળવે છે"""
    if not os.path.exists(BASE_STORAGE_DIR):
        return []
    return [d for d in os.listdir(BASE_STORAGE_DIR) if os.path.isdir(os.path.join(BASE_STORAGE_DIR, d))]

# ============================================================
# HELPER FUNCTIONS (Events)
# ============================================================
def get_events_list():
    """Drive પરથી બધી ઇવેન્ટ્સ શોધો"""
    try:
        drive_service = get_drive_service()
        query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])
        return [f['name'] for f in folders if f['name'] not in ['images', 'temp_crops']]
    except:
        return []

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
# CUSTOM CSS (Mobile Responsive)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    .logo-area {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .logo-area img {
        height: 55px;
        width: auto;
        border-radius: 12px;
    }
    .brand-text h1 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f0f0f;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .brand-text h1 span {
        color: #d4af37;
    }
    .brand-text .tagline {
        font-size: 0.85rem;
        font-weight: 400;
        color: #6c757d;
        margin: -5px 0 0 0;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 100%);
        padding: 2rem 1rem;
    }
    .sidebar-logo {
        text-align: center;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 1.5rem;
    }
    .sidebar-logo img {
        width: 80%;
        max-width: 180px;
        border-radius: 16px;
        background: white;
        padding: 8px;
        margin-bottom: 10px;
    }
    .sidebar-logo .brand-name {
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        margin: 0;
    }
    .sidebar-logo .brand-name span {
        color: #d4af37;
    }
    .card {
        background: white;
        border: 1px solid #f0f0f0;
        border-radius: 24px;
        padding: 1.8rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1.5rem;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.08);
    }
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f0f0f;
        margin-bottom: 0.3rem;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #6c757d;
        line-height: 1.6;
    }
    .stButton button {
        background: linear-gradient(135deg, #0f0f0f 0%, #333333 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2.2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25) !important;
        background: linear-gradient(135deg, #1a1a1a 0%, #444444 100%) !important;
    }
    .footer {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        margin-top: 3rem;
        border-top: 1px solid #f0f0f0;
        color: #adb5bd;
        font-size: 0.8rem;
        font-weight: 400;
    }
    .footer strong {
        color: #0f0f0f;
        font-weight: 700;
    }
    .footer span {
        color: #d4af37;
    }
    @media (max-width: 768px) {
        .logo-area img { height: 40px !important; }
        .brand-text h1 { font-size: 1.5rem !important; }
        .brand-text .tagline { font-size: 0.7rem !important; }
        .main-header { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
        .card { padding: 1rem !important; border-radius: 16px !important; }
        .card-title { font-size: 1.1rem !important; }
        .card-desc { font-size: 0.85rem !important; }
        .stButton button { font-size: 0.8rem !important; padding: 0.4rem 1.2rem !important; width: 100% !important; }
        .stImage { width: 100% !important; }
        section[data-testid="stSidebar"] { padding: 0.5rem !important; }
        .sidebar-logo img { max-width: 120px !important; }
        .sidebar-logo .brand-name { font-size: 1rem !important; }
        .stSidebar .stColumns { gap: 0.3rem !important; }
        .stSidebar .stButton button { font-size: 0.7rem !important; padding: 0.3rem 0.6rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER & SIDEBAR
# ============================================================
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
# PASSWORD PROTECTION
# ============================================================
def check_password():
    if st.session_state.get("authenticated", False):
        return True
    
    st.sidebar.markdown("---")
    password = st.sidebar.text_input("🔒 એડમિન પાસવર્ડ:", type="password", key="admin_pass")
    
    if password:
        # સીધો પાસવર્ડ ચેક
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
# TELEGRAM BOT
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

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

app = load_insightface()

# ============================================================
# CONSTANTS
# ============================================================
PHOTO_PRICE = 10
# ============================================================
# PAGE 1: MANAGE EVENTS
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
            st.write("🔍 DEBUG: Button clicked!")
            if new_event.strip() and event_password.strip():
                event_name = new_event.strip()
                if event_name:
                    event_path, photos_path = get_event_dir(event_name)
                    
                    # નવી ઇવેન્ટ માટે ખાલી ડેટા ફાઇલ બનાવો
                    initial_data = {"password": event_password, "faces": []}
                    if save_event_data_local(event_name, initial_data):
                        st.success(f"✅ ઇવેન્ટ '{event_name}' સફળતાપૂર્વક બની ગઈ!")
                        st.rerun()
                    else:
                        st.error("❌ ઇવેન્ટ બનાવવામાં ભૂલ આવી.")
            else:
                st.error("❌ કૃપા કરીને નામ અને પાસવર્ડ બંને ભરો.")

    # બધી ઇવેન્ટ્સનું લિસ્ટ મેળવો
    available_events = list_all_local_events()

    if not available_events:
        st.info("ℹ️ હજુ સુધી કોઈ ઇવેન્ટ નથી. ઉપર નવી ઇવેન્ટ બનાવો.")
    else:
        # ઇવેન્ટ સિલેક્ટ કરવાનો ડ્રોપડાઉન
        selected_event = st.selectbox("📁 ઇવેન્ટ પસંદ કરો", available_events)
        
        if selected_event:
            st.subheader(f"📸 ફોટા અપલોડ કરો - {selected_event}")
            
            # ફોટો અપલોડ બટન
            uploaded_files = st.file_uploader(
                "ઇવેન્ટના ફોટા પસંદ કરો", 
                type=['jpg', 'jpeg', 'png'], 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("🚀 ફોટા પ્રોસેસ અને સેવ કરો"):
                    event_path, photos_path = get_event_dir(selected_event.strip())

                    # જો ફોલ્ડર ન બન્યું હોય તો એરર બતાવો
                    if not os.path.exists(photos_path):
                        st.error("❌ ઇવેન્ટ ફોલ્ડર મળ્યું નહીં!")
                        st.stop()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_files = len(uploaded_files)
                    
                    # ઇવેન્ટનો લોકલ ડેટા લોડ કરો
                    event_data = load_event_data_local(selected_event.strip())
                    existing_faces = event_data.get("faces", [])
                    processed_count = 0
                    
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"⏳ {file.name} પર કામ ચાલુ છે... ({i+1}/{total_files})")
                        
                        # ૧. ફોટો લોકલ ડિરેક્ટરીમાં સેવ કરો
                        file_path = os.path.join(photos_path, file.name)
                        file.seek(0)
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                        
                        # ૨. સેવ કરેલા ફોટામાંથી ફેસ ડિટેક્શન કરો
                        img = cv2.imread(file_path)
                        if img is None:
                            st.warning(f"⚠️ {file.name} વાંચવામાં ભૂલ આવી.")
                            continue
                            
                        faces = app.get(img)
                        
                        if len(faces) == 0:
                            st.warning(f"⚠️ {file.name} માં કોઈ ચહેરો મળ્યો નથી.")
                            continue
                        
                        # ૩. દરેક મળેલા ચહેરાના એમ્બેડિંગ્સ લિસ્ટમાં ઉમેરો
                        for face_idx, face in enumerate(faces):
                            embedding_list = face.embedding.tolist()
                            existing_faces.append({
                                "photo_name": file.name,
                                "file_path": file_path,
                                "face_index": face_idx,
                                "embedding": embedding_list
                            })
                            
                        processed_count += 1
                        progress_bar.progress((i + 1) / total_files)
                    
                    # ૪. ઇવેન્ટનો અપડેટ થયેલો ડેટા JSON માં સેવ કરો
                    event_data["faces"] = existing_faces
                    save_event_data_local(selected_event.strip(), event_data)
                    
                    status_text.empty()
                    st.success(f"✅ {processed_count} ફોટા સફળતાપૂર્વક પ્રોસેસ અને સેવ થઈ ગયા!")
                    st.rerun()
                    
                    # ડ્રાઈવ પર અપલોડ
                    file_ext = os.path.splitext(file.name)[1]
                    unique_name = f"{hashlib.md5(file.name.encode() + str(datetime.datetime.now()).encode()).hexdigest()[:10]}{file_ext}"
                    # લૂપની અંદર:
                    file_path = os.path.join(photos_path, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if drive_file_id is None:
                        st.error(f"❌ {file.name} Drive પર અપલોડ થયો નહીં!")
                        continue
                    
                    # દરેક ચહેરા માટે
                    for j, face in enumerate(faces):
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        pad = 20
                        h, w = img.shape[:2]
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        
                        face_crop = img[y1:y2, x1:x2]
                        crop_filename = f"{hashlib.md5(unique_name.encode() + str(j).encode()).hexdigest()[:8]}.jpg"
                        crop_path = os.path.join("temp_crops", crop_filename)
                        os.makedirs("temp_crops", exist_ok=True)
                        cv2.imwrite(crop_path, face_crop)
                        
                        embedding = face.embedding / np.linalg.norm(face.embedding)                       
                        # --- SMART AUTO-LABEL CHECK ---
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
                        
                        # જો પહેલેથી જાણીતો ચહેરો હોય તો સીધો સેવ કરી દો (ફરી પૂછવું નહીં પડે)
                        if matched_label and matched_label != "SKIP":
                            existing_faces.append({
                                "filename": unique_name,
                                "drive_file_id": drive_file_id,
                                "person_label": matched_label,
                                "embedding": embedding.tolist()
                            })
                            auto_saved_count += 1
                            # ક્રોપ કરેલી ફાઈલ ડીલીટ કરી દો
                            if os.path.exists(crop_path):
                                os.remove(crop_path)
                        else:
                            # જો નવો ચહેરો હોય તો જ યુઝરને લેબલ પૂછવા માટે pending_faces માં ઉમેરો
                            st.session_state.pending_faces.append({
                                "crop_path": crop_path,
                                "embedding": embedding.tolist(),
                                "original_filename": unique_name,
                                "drive_file_id": drive_file_id,
                                "label": "SKIP"
                            })
                    
                    progress_bar.progress((i + 1) / total_files)
                
                # ઓટો-મેચ થયેલા ચહેરાઓ ડ્રાઈવ/ડેટાબેઝમાં અપડેટ કરો
                if auto_saved_count > 0:
                    event_data["faces"] = existing_faces
                    save_event_data(selected_event, event_data, folder_id)
                    st.info(f"✨ {auto_saved_count} ચહેરાઓ આપોઆપ ઓળખાઈને સેવ થઈ ગયા!")
                
                status_text.text("✅ પ્રોસેસ પૂર્ણ થઈ!")
            
            # ---------- SMART GROUP LABELING (ફક્ત નવા ન ઓળખાયેલા ચહેરાઓ માટે) ----------
            if 'pending_faces' in st.session_state and st.session_state.pending_faces:
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
                    event_data = load_event_data(selected_event)
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
                    folder_id = get_drive_folder_id(selected_event)
                    save_event_data(selected_event, event_data, folder_id)
                    
                    for face_data in pending:
                        try:
                            if os.path.exists(face_data["crop_path"]):
                                os.remove(face_data["crop_path"])
                        except:
                            pass
                    st.session_state.pending_faces = []
                    st.cache_resource.clear()
                    st.success(f"✅ {count} નવા ચહેરા '{selected_event}' માં Drive પર સેવ થયા!")
                    st.rerun()
            
            st.divider()
            event_data = load_event_data(selected_event)
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
                                st.write(f"❌ {item['filename']} (Drive ID missing)")
            else:
                st.info("ℹ️ હજુ સુધી કોઈ ફોટો લેબલ થયો નથી.")
            
            # ===== DELETE EVENT =====
            st.divider()
            st.markdown("### 🗑️ ઇવેન્ટ કાઢી નાખો")
            st.warning(f"⚠️ આ ઇવેન્ટ ('{selected_event}') અને તેના બધા Drive ફોટા કાયમ માટે ડિલીટ થઈ જશે!")
            if st.button(f"🗑️ '{selected_event}' ઇવેન્ટ કાઢી નાખો", type="primary"):
                try:
                    drive_service = get_drive_service()
                    folder_id = get_drive_folder_id(selected_event)
                    if folder_id:
                        drive_service.files().delete(fileId=folder_id).execute()
                    st.success(f"✅ '{selected_event}' ઇવેન્ટ Drive પરથી ડિલીટ થઈ ગઈ!")
                    st.session_state.pending_faces = []
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ ઇવેન્ટ ડિલીટ કરતી વખતે ભૂલ: {e}")

# ============================================================
# PAGE 2: SEARCH FACE
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
                    event_data = load_event_data(event_name)
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
                    uploaded_file = st.file_uploader(
                        "📁 ફોટો પસંદ કરો...",
                        type=["jpg", "jpeg", "png"],
                        key="file_uploader"
                    )
                
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
                                    query_embeddings.append({
                                        "query_face": f"face_{i}",
                                        "embedding": emb.tolist()
                                    })
                                
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
                                    # TELEGRAM NOTIFICATION
                                    send_telegram_message(
                                        f"✅ <b>નવો ગ્રાહક!</b>\n"
                                        f"📸 ઇવેન્ટ: {event_name}\n"
                                        f"👤 વ્યક્તિ: {', '.join(matched_persons)}\n"
                                        f"🕒 {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}"
                                    )
                                    
                                    # CART SYSTEM
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
                                                        img_path = None
                                                        if file_id:
                                                            img_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                                                            st.image(img_url, width=150)
                                                            img_path = img_url
                                                        else:
                                                            img_path = os.path.join("events", event_name, "images", item["filename"])
                                                            st.image(img_path, width=150)
                                                    except Exception as e:
                                                        st.write(f"📁 {item.get('filename', 'Unknown')}")
                                                        img_path = None

                                                    price = PHOTO_PRICE
                                                    cart_key = f"cart_{person}_{idx}"
                                                    selected = st.checkbox(f"🛒 ₹{price}" if price > 0 else "🆓 FREE", key=cart_key)
                                                    
                                                    if selected:
                                                        if "cart" not in st.session_state:
                                                            st.session_state.cart = []
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
                                                        if "cart" in st.session_state:
                                                            st.session_state.cart = [c for c in st.session_state.cart if not (c["person"] == person and c["filename"] == item["filename"])]
                                            
                                            if st.button(f"➕ {person} ના બધા ફોટા કાર્ટમાં ઉમેરો", key=f"add_all_{person}"):
                                                for item in person_photos:
                                                    img_path = os.path.join("events", event_name, "images", item["filename"])
                                                    price = PHOTO_PRICE
                                                    cart_item = {
                                                        "person": person,
                                                        "filename": item["filename"],
                                                        "price": price,
                                                        "img_path": img_path
                                                    }
                                                    if "cart" not in st.session_state:
                                                        st.session_state.cart = []
                                                    if cart_item not in st.session_state.cart:
                                                        st.session_state.cart.append(cart_item)
                                                st.rerun()
                                        else:
                                            st.write("❌ આ વ્યક્તિના કોઈ ફોટા નથી.")
                                    
                                    # CART DISPLAY
                                    st.sidebar.markdown("---")
                                    st.sidebar.markdown("## 🛒 તમારું કાર્ટ")
                                    
                                    if "cart" in st.session_state and st.session_state.cart:
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
                                        
                                        # CHECKOUT
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
                                            st.sidebar.link_button("🟢 Google Pay", upi_links["📱 Google Pay"], width="stretch")
                                            st.sidebar.link_button("🟠 PhonePe", upi_links["📱 PhonePe"], width="stretch")
                                            st.sidebar.link_button("🔵 Paytm", upi_links["📱 Paytm"], width="stretch")
                                            st.sidebar.link_button("🟣 BHIM", upi_links["📱 BHIM"], width="stretch")
                                            st.sidebar.link_button("📱 અન્ય UPI એપ", upi_links["📱 Generic UPI"], width="stretch")
                                            
                                            st.sidebar.markdown("---")
                                            if st.sidebar.button("✅ પેમેન્ટ થઈ ગયું!", width="stretch"):
                                                unique_persons = set()
                                                for item in cart:
                                                    unique_persons.add(item['person'])
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
                                        
                                        # PAYMENT DONE -> DOWNLOAD
                                        if st.session_state.get("payment_done", False):
                                            st.sidebar.markdown("---")
                                            st.sidebar.markdown("## 📥 તમારા ફોટા ડાઉનલોડ કરો")
                                            
                                            if st.sidebar.button("📥 બધા ફોટા ડાઉનલોડ કરો (ZIP)"):
                                                import zipfile
                                                import io
                                                zip_buffer = io.BytesIO()
                                                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                                                    for item in cart:
                                                        img_path = item["img_path"]
                                                        if os.path.exists(img_path):
                                                            zip_file.write(img_path, item["filename"])
                                                zip_buffer.seek(0)
                                                st.sidebar.download_button(
                                                    label="📥 ZIP ડાઉનલોડ કરો",
                                                    data=zip_buffer,
                                                    file_name="my_photos.zip",
                                                    mime="application/zip",
                                                    key="zip_download_final"
                                                )
                                            
                                            for idx, item in enumerate(cart):
                                                img_path = item["img_path"]
                                                if os.path.exists(img_path):
                                                    with open(img_path, "rb") as f:
                                                        st.sidebar.download_button(
                                                            label=f"📸 {item['filename']} ડાઉનલોડ કરો",
                                                            data=f,
                                                            file_name=item["filename"],
                                                            mime="image/jpeg",
                                                            key=f"final_dl_{idx}"
                                                        )
                                            
                                            st.sidebar.markdown("---")
                                            st.sidebar.markdown("## 📤 તમારા ફોટા શેર કરો")
                                            
                                            app_url = "https://jayphotofinder.streamlit.app"
                                            share_text = "🌟 મારા ઇવેન્ટના સુંદર ફોટા જુઓ! જય ફોટો શોધ દ્વારા શોધ્યા."
                                            
                                            whatsapp_url = f"https://api.whatsapp.com/send?text={share_text} {app_url}"
                                            st.sidebar.markdown(f"[![WhatsApp](https://img.icons8.com/color/48/000000/whatsapp.png)]({whatsapp_url}) શેર કરો")
                                            
                                            facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={app_url}"
                                            st.sidebar.markdown(f"[![Facebook](https://img.icons8.com/color/48/000000/facebook.png)]({facebook_url}) શેર કરો")
                                            
                                            st.sidebar.markdown("📸 **Instagram:** લિંક કોપી કરીને પેસ્ટ કરો")
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
# PAGE 3: GENERATE QR CODE
# ============================================================
elif option == "📱 QR કોડ બનાવો":
    st.markdown("""
    <div class="card">
        <div class="card-title">📱 QR કોડ બનાવો</div>
        <div class="card-desc">અહીં તમે કોઈ પણ ઇવેન્ટ માટે QR કોડ બનાવી શકો છો. ગ્રાહકો આ QR કોડ સ્કેન કરીને તેમના ફોટા શોધી શકશે.</div>
    </div>
    """, unsafe_allow_html=True)
    
    events = get_events_list()
    
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
# PAGE 4: BENCHMARK
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
# 📌 FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    📸 <strong>જય ફોટો શોધ</strong> - AI દ્વારા તમારા ફોટા શોધો<br>
    © 2026 Jay Photography | Made with ❤️ in Gujarat
</div>
""", unsafe_allow_html=True)