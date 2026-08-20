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
import zipfile
import io
from io import BytesIO
from insightface.app import FaceAnalysis
from face_search import find_best_global_assignment
from PIL import Image

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
BASE_STORAGE_DIR = "events_data"
os.makedirs(BASE_STORAGE_DIR, exist_ok=True)

def get_event_dir(event_name):
    """ઇવેન્ટ માટે લોકલ ફોલ્ડર બનાવે છે"""
    event_path = os.path.join(BASE_STORAGE_DIR, event_name)
    photos_path = os.path.join(event_path, "photos")
    crops_path = os.path.join(event_path, "crops")
    os.makedirs(photos_path, exist_ok=True)
    os.makedirs(crops_path, exist_ok=True)
    return event_path, photos_path, crops_path

def save_event_data_local(event_name, data):
    """ઇવેન્ટનો ડેટા અને ફેસ એમ્બેડિંગ્સ JSON માં સેવ કરે છે"""
    try:
        event_path, _, _ = get_event_dir(event_name)
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
        event_path, _, _ = get_event_dir(event_name)
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
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .brand-text h1 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f0f0f;
        margin: 0;
    }
    .brand-text h1 span { color: #d4af37; }
    .card {
        background: white;
        border: 1px solid #f0f0f0;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
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
    }
    .footer {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        margin-top: 3rem;
        border-top: 1px solid #f0f0f0;
        color: #adb5bd;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER & SIDEBAR
# ============================================================
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("assets/logo.jpg"):
        st.image("assets/logo.jpg", width=100)
    else:
        st.markdown("## 📸")
with col2:
    st.markdown("""
    <div class="brand-text">
        <h1 style="font-size: 2.5rem; font-weight: 900;">
            JAY <span style="color: #d4af37;">PHOTO</span> SHODH
        </h1>
        <div style="color: #6c757d; font-size: 0.9rem; letter-spacing: 2px;">
            ✨ AI POWERED PHOTO SEARCH
        </div>
    </div>
    """, unsafe_allow_html=True)

if os.path.exists("assets/logo.jpg"):
    st.sidebar.image("assets/logo.jpg", use_container_width=True)

option = st.sidebar.selectbox(
    "📌 પેજ પસંદ કરો",
    ["🔍 ફોટો શોધો", "📂 ઇવેન્ટ મેનેજ", "📱 QR કોડ બનાવો", "📊 Analytics", "📊 બેન્ચમાર્ક"]
)

# ============================================================
# TELEGRAM BOT
# ============================================================
def send_telegram_message(message):
    try:
        TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
        TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
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

app = load_insightface()
PHOTO_PRICE = 10

# ============================================================
# PAGE 1: MANAGE EVENTS
# ============================================================
if option == "📂 ઇવેન્ટ મેનેજ":
    st.markdown("""
    <div class="card">
        <div class="card-title">📂 ઇવેન્ટ મેનેજમેન્ટ</div>
        <div class="card-desc">અહીં તમે નવી ઇવેન્ટ બનાવી શકો છો, ફોટા અપલોડ કરી શકો છો અને ચહેરાઓને મેનેજ કરી શકો છો.</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("➕ નવી ઇવેન્ટ બનાવો", expanded=False):
        new_event = st.text_input("ઇવેન્ટનું નામ (દા.ત., sharma_wedding)")
        event_password = st.text_input("🔒 ઇવેન્ટ પાસવર્ડ (ગ્રાહકો માટે)", type="password")
        
        if st.button("📌 ઇવેન્ટ બનાવો", key="create_event"):
            if new_event.strip() and event_password.strip():
                clean_name = new_event.strip().replace(" ", "_")
                get_event_dir(clean_name)
                initial_data = {"password": event_password.strip(), "faces": []}
                if save_event_data_local(clean_name, initial_data):
                    st.success(f"✅ ઇવેન્ટ '{clean_name}' સફળતાપૂર્વક બની ગઈ!")
                    st.rerun()
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
                    event_path, photos_path, crops_path = get_event_dir(selected_event.strip())
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_files = len(uploaded_files)
                    
                    event_data = load_event_data_local(selected_event.strip())
                    existing_faces = event_data.get("faces", [])
                    processed_count = 0
                    
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"⏳ {file.name} પર કામ ચાલુ છે... ({i+1}/{total_files})")
                        
                        file_path = os.path.join(photos_path, file.name)
                        file.seek(0)
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                        
                        img = cv2.imread(file_path)
                        if img is None:
                            st.warning(f"⚠️ {file.name} વાંચવામાં ભૂલ આવી.")
                        else:
                            faces = app.get(img)
                            if len(faces) == 0:
                                st.warning(f"⚠️ {file.name} માં કોઈ ચહેરો મળ્યો નથી.")
                            else:
                                for face_idx, face in enumerate(faces):
                                    norm_emb = face.embedding / np.linalg.norm(face.embedding)
                                    existing_faces.append({
                                        "photo_name": file.name,
                                        "file_path": file_path,
                                        "face_index": face_idx,
                                        "person_label": "Guest",
                                        "embedding": norm_emb.tolist()
                                    })
                                processed_count += 1
                        
                        progress_bar.progress((i + 1) / total_files)
                    
                    event_data["faces"] = existing_faces
                    save_event_data_local(selected_event.strip(), event_data)
                    st.cache_resource.clear()
                    status_text.empty()
                    st.success(f"✅ {processed_count} ફોટા સફળતાપૂર્વક પ્રોસેસ અને સેવ થઈ ગયા!")
                    st.rerun()
            
            st.divider()
            event_data = load_event_data(selected_event)
            faces_list = event_data.get("faces", [])
            st.write(f"📊 આ ઇવેન્ટમાં કુલ **{len(faces_list)}** ઓળખાયેલા ચહેરાઓ છે.")
            
            # Delete Event
            st.divider()
            st.markdown("### 🗑️ ઇવેન્ટ કાઢી નાખો")
            if st.button(f"🗑️ '{selected_event}' ઇવેન્ટ ડિલીટ કરો", type="primary"):
                event_path, _, _ = get_event_dir(selected_event)
                if os.path.exists(event_path):
                    shutil.rmtree(event_path)
                st.cache_resource.clear()
                st.success(f"✅ '{selected_event}' ઇવેન્ટ ડિલીટ થઈ ગઈ!")
                st.rerun()

# ============================================================
# PAGE 2: SEARCH FACE
# ============================================================
elif option == "🔍 ફોટો શોધો":
    query_params = st.query_params
    event_name = query_params.get("event", None)
    
    if event_name is None:
        available_events = list_all_local_events()
        if available_events:
            event_name = st.selectbox("📂 ઇવેન્ટ પસંદ કરો", available_events)
        else:
            st.info("ℹ️ હજુ સુધી કોઈ ઇવેન્ટ ઉપલબ્ધ નથી.")
            st.stop()
    
    event_path, photos_path, _ = get_event_dir(event_name)
    if not os.path.exists(event_path):
        st.error(f"❌ '{event_name}' ઇવેન્ટ મળી નહીં.")
    else:
        if f"auth_{event_name}" not in st.session_state:
            st.session_state[f"auth_{event_name}"] = False
        
        event_data = load_event_data(event_name)
        
        # Password Lock
        if event_data.get("password") and not st.session_state[f"auth_{event_name}"]:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🔒 '{event_name}' ઇવેન્ટ માટે પાસવર્ડ</div>
            </div>
            """, unsafe_allow_html=True)
            entered_password = st.text_input("🔑 પાસવર્ડ:", type="password")
            if st.button("🚪 પ્રવેશ કરો"):
                if event_data.get("password") == entered_password:
                    st.session_state[f"auth_{event_name}"] = True
                    st.success("✅ પ્રવેશ મળ્યો!")
                    st.rerun()
                else:
                    st.error("❌ ખોટો પાસવર્ડ!")
            st.stop()
        
        st.subheader("📸 ફોટો અપલોડ કરો અથવા સેલ્ફી લો")
        upload_option = st.radio("રીત પસંદ કરો:", ["📸 કેમેરાથી સેલ્ફી લો", "📁 ફોટો અપલોડ કરો"], horizontal=True)
        
        uploaded_file = st.camera_input("📸 સેલ્ફી લો") if upload_option == "📸 કેમેરાથી સેલ્ફી લો" else st.file_uploader("📁 ફોટો પસંદ કરો", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            img = cv2.imread(tmp_path)
            if img is not None:
                st.image(img, channels="BGR", caption="તમારો ફોટો", width=250)
                with st.spinner("🔍 ચહેરો શોધાઈ રહ્યો છે..."):
                    faces = app.get(img)
                    if len(faces) == 0:
                        st.warning("❌ કોઈ ચહેરો ઓળખાયો નહીં.")
                    else:
                        query_emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
                        matched_photos = []
                        
                        # Match with all event photos
                        for item in event_data.get("faces", []):
                            db_emb = parse_embedding(item.get("embedding"))
                            if db_emb is not None:
                                sim = float(np.dot(query_emb, db_emb))
                                if sim >= 0.45: # Similarity threshold
                                    matched_photos.append((item["file_path"], item["photo_name"], sim))
                        
                        # Remove duplicate photos
                        seen = set()
                        unique_matches = []
                        for p_path, p_name, sim in matched_photos:
                            if p_name not in seen and os.path.exists(p_path):
                                seen.add(p_name)
                                unique_matches.append((p_path, p_name, sim))
                        
                        if unique_matches:
                            st.success(f"🎉 તમારા {len(unique_matches)} ફોટા મળ્યા!")
                            
                            cols = st.columns(3)
                            for idx, (p_path, p_name, sim) in enumerate(unique_matches):
                                col = cols[idx % 3]
                                with col:
                                    st.image(p_path, caption=f"{p_name}", use_container_width=True)
                                    with open(p_path, "rb") as f:
                                        st.download_button(
                                            label=f"⬇️ ડાઉનલોડ",
                                            data=f,
                                            file_name=p_name,
                                            mime="image/jpeg",
                                            key=f"dl_{idx}"
                                        )
                        else:
                            st.warning("🔍 આ ઇવેન્ટમાં તમારો કોઈ મેળ ખાતો ફોટો મળ્યો નથી.")

# ============================================================
# PAGE 3: GENERATE QR CODE
# ============================================================
elif option == "📱 QR કોડ બનાવો":
    st.markdown("""
    <div class="card">
        <div class="card-title">📱 QR કોડ બનાવો</div>
        <div class="card-desc">ગ્રાહકો માટે સીધા ઇવેન્ટ પર જવા માટેનો QR કોડ અહીંથી બનાવો.</div>
    </div>
    """, unsafe_allow_html=True)
    
    events = list_all_local_events()
    if not events:
        st.warning("⚠️ હજુ સુધી કોઈ ઇવેન્ટ નથી.")
    else:
        selected_event = st.selectbox("📂 ઇવેન્ટ પસંદ કરો", events)
        if selected_event:
            # તમારી લાઈવ Streamlit URL અહીં સેટ કરો
            url = f"https://face-photo-finder.streamlit.app/?event={selected_event}"
            qr_img = qrcode.make(url)
            
            buf = BytesIO()
            qr_img.save(buf, format="PNG")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(buf.getvalue(), width=250, caption=f"{selected_event} QR Code")
                st.download_button("⬇️ QR કોડ ડાઉનલોડ કરો", data=buf.getvalue(), file_name=f"{selected_event}_qr.png", mime="image/png")
            with col2:
                st.info("💡 ગ્રાહક આ QR સ્કેન કરશે એટલે સીધી આ ઇવેન્ટ ખુલશે.")
                st.code(url)

# ============================================================
# PAGE 4: ANALYTICS & BENCHMARK
# ============================================================
elif option == "📊 Analytics":
    st.header("📊 વપરાશ એનાલિટિક્સ")
    if os.path.exists("analytics.csv"):
        df = pd.read_csv("analytics.csv")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("ℹ️ હજુ કોઈ એનાલિટિક્સ ડેટા ઉપલબ્ધ નથી.")

else:
    st.header("📊 બેન્ચમાર્ક પરિણામો")
    if os.path.exists("benchmark_results.csv"):
        df = pd.read_csv("benchmark_results.csv")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("ℹ️ benchmark_results.csv ફાઇલ મળી નથી.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    📸 <strong>જય ફોટો શોધ</strong> - AI દ્વારા તમારા ફોટા શોધો<br>
    © 2026 Jay Photography | Made with ❤️ in Gujarat
</div>
""", unsafe_allow_html=True)