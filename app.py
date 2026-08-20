import streamlit as st
import cv2
import numpy as np
import json
import qrcode
import os
import shutil
import datetime
import tempfile
import faiss
import requests
import urllib.parse
import csv
import pandas as pd
from io import BytesIO
from insightface.app import FaceAnalysis
from PIL import Image

# Session state initialization
if "pending_faces" not in st.session_state:
    st.session_state.pending_faces = []

# ============================================================
# ENVIRONMENT VARIABLE
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
# LOCAL STORAGE SYSTEM
# ============================================================
BASE_STORAGE_DIR = "events_data"
os.makedirs(BASE_STORAGE_DIR, exist_ok=True)

def get_event_dir(event_name):
    clean_name = event_name.strip().replace(" ", "_")
    event_path = os.path.join(BASE_STORAGE_DIR, clean_name)
    photos_path = os.path.join(event_path, "photos")
    crops_path = os.path.join(event_path, "crops")
    os.makedirs(photos_path, exist_ok=True)
    os.makedirs(crops_path, exist_ok=True)
    return event_path, photos_path, crops_path

def save_event_data_local(event_name, data):
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
    try:
        event_path, _, _ = get_event_dir(event_name)
        data_file = os.path.join(event_path, f"{event_name}_data.json")
        if os.path.exists(data_file):
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"password": "", "faces": []}
    except Exception as e:
        return {"password": "", "faces": []}

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
    if embedding_data is None:
        return None
    if isinstance(embedding_data, list):
        return np.array(embedding_data, dtype=np.float32)
    if isinstance(embedding_data, np.ndarray):
        return embedding_data
    return None

app = load_insightface()
PHOTO_PRICE = 10

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
    <div style="margin-top: 10px;">
        <h1 style="font-size: 2.2rem; font-weight: 900; margin:0;">
            JAY <span style="color: #d4af37;">PHOTO</span> SHODH
        </h1>
        <div style="color: #6c757d; font-size: 0.85rem; letter-spacing: 2px;">
            ✨ AI POWERED PHOTO SEARCH
        </div>
    </div>
    """, unsafe_allow_html=True)

option = st.sidebar.selectbox(
    "📌 પેજ પસંદ કરો",
    ["🔍 ફોટો શોધો", "📂 ઇવેન્ટ મેનેજ", "📱 QR કોડ બનાવો"]
)

# ============================================================
# PAGE 1: MANAGE EVENTS & SMART LABELING
# ============================================================
if option == "📂 ઇવેન્ટ મેનેજ":
    st.subheader("📂 ઇવેન્ટ મેનેજમેન્ટ")
    
    with st.expander("➕ નવી ઇવેન્ટ બનાવો", expanded=False):
        new_event = st.text_input("ઇવેન્ટનું નામ (દા.ત., sharma_wedding)")
        event_password = st.text_input("🔒 ઇવેન્ટ પાસવર્ડ (ગ્રાહકો માટે)", type="password")
        
        if st.button("📌 ઇવેન્ટ બનાવો"):
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
            st.markdown(f"### 📸 ફોટા અપલોડ કરો - `{selected_event}`")
            
            uploaded_files = st.file_uploader(
                "ઇવેન્ટના ફોટા પસંદ કરો", 
                type=['jpg', 'jpeg', 'png'], 
                accept_multiple_files=True
            )
            
            if uploaded_files and st.button("🚀 ફોટા પ્રોસેસ અને ચહેરા શોધો"):
                event_path, photos_path, crops_path = get_event_dir(selected_event)
                event_data = load_event_data_local(selected_event)
                existing_faces = event_data.get("faces", [])
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_files = len(uploaded_files)
                new_pending = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"⏳ {file.name} પ્રોસેસ થઈ રહ્યો છે... ({i+1}/{total_files})")
                    
                    file_path = os.path.join(photos_path, file.name)
                    file.seek(0)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    img = cv2.imread(file_path)
                    if img is not None:
                        faces = app.get(img)
                        h, w = img.shape[:2]
                        
                        for j, face in enumerate(faces):
                            norm_emb = face.embedding / np.linalg.norm(face.embedding)
                            
                            # ચહેરો ક્રોપ કરો
                            bbox = face.bbox.astype(int)
                            x1, y1, x2, y2 = max(0, bbox[0]-15), max(0, bbox[1]-15), min(w, bbox[2]+15), min(h, bbox[3]+15)
                            crop = img[y1:y2, x1:x2]
                            crop_name = f"crop_{i}_{j}_{file.name}"
                            crop_file_path = os.path.join(crops_path, crop_name)
                            cv2.imwrite(crop_file_path, crop)
                            
                            # Auto-match check
                            matched_label = None
                            for ef in existing_faces:
                                db_emb = parse_embedding(ef.get("embedding"))
                                if db_emb is not None and float(np.dot(norm_emb, db_emb)) > 0.65:
                                    matched_label = ef.get("person_label")
                                    break
                            
                            if matched_label and matched_label != "SKIP":
                                existing_faces.append({
                                    "photo_name": file.name,
                                    "file_path": file_path,
                                    "person_label": matched_label,
                                    "embedding": norm_emb.tolist()
                                })
                            else:
                                new_pending.append({
                                    "crop_path": crop_file_path,
                                    "file_path": file_path,
                                    "photo_name": file.name,
                                    "embedding": norm_emb.tolist(),
                                    "label": ""
                                })
                    progress_bar.progress((i + 1) / total_files)
                
                event_data["faces"] = existing_faces
                save_event_data_local(selected_event, event_data)
                st.session_state.pending_faces = new_pending
                st.cache_resource.clear()
                status_text.empty()
                st.success("✅ ફોટા સ્કેન થઈ ગયા! નીચે ચહેરાઓને નામ આપો.")
                st.rerun()
            
            # ---------- SMART GROUP LABELING UI ----------
            if st.session_state.pending_faces:
                st.divider()
                st.subheader(f"🏷️ {len(st.session_state.pending_faces)} નવા ચહેરાઓને નામ આપો (Smart Groups)")
                pending = st.session_state.pending_faces
                
                # કલસ્ટરિંગ (સરખા ચહેરાઓ ભેગા કરવા)
                embs = np.array([f["embedding"] for f in pending], dtype=np.float32)
                sim_matrix = np.dot(embs, embs.T)
                n = len(pending)
                visited = [False] * n
                clusters = []
                for i in range(n):
                    if not visited[i]:
                        cl = [i]
                        visited[i] = True
                        for j in range(i+1, n):
                            if not visited[j] and sim_matrix[i][j] > 0.65:
                                cl.append(j)
                                visited[j] = True
                        clusters.append(cl)
                
                # દરેક ગ્રૂપ માટે UI
                for g_idx, cl in enumerate(clusters):
                    st.markdown(f"**🎯 ગ્રૂપ {g_idx + 1} ({len(cl)} ચહેરા)**")
                    cols = st.columns(min(5, len(cl)))
                    for c_i, f_idx in enumerate(cl[:5]):
                        with cols[c_i]:
                            if os.path.exists(pending[f_idx]["crop_path"]):
                                st.image(pending[f_idx]["crop_path"], width=100)
                    
                    lbl = st.text_input(f"આ વ્યક્તિનું નામ લખો (દા.ત. વરરાજા, દુલહન, રમેશભાઈ):", key=f"grp_lbl_{g_idx}")
                    for f_idx in cl:
                        pending[f_idx]["label"] = lbl.strip() if lbl.strip() else "Guest"
                    st.write("---")
                
                if st.button("💾 બધા નામો સેવ કરો"):
                    event_data = load_event_data_local(selected_event)
                    existing_faces = event_data.get("faces", [])
                    for pf in pending:
                        existing_faces.append({
                            "photo_name": pf["photo_name"],
                            "file_path": pf["file_path"],
                            "person_label": pf["label"],
                            "embedding": pf["embedding"]
                        })
                    event_data["faces"] = existing_faces
                    save_event_data_local(selected_event, event_data)
                    st.session_state.pending_faces = []
                    st.cache_resource.clear()
                    st.success("✅ બધા ચહેરા સફળતાપૂર્વક લેબલ થઈ ગયા!")
                    st.rerun()

            # લિસ્ટિંગ અને ડિલીટ
            st.divider()
            event_data = load_event_data_local(selected_event)
            faces_list = event_data.get("faces", [])
            st.write(f"📊 આ ઇવેન્ટમાં કુલ **{len(faces_list)}** ઓળખાયેલા ચહેરાઓ છે.")
            
            if st.button(f"🗑️ '{selected_event}' ઇવેન્ટ ડિલીટ કરો", type="primary"):
                event_path, _, _ = get_event_dir(selected_event)
                if os.path.exists(event_path):
                    shutil.rmtree(event_path)
                st.cache_resource.clear()
                st.success("✅ ઇવેન્ટ ડિલીટ થઈ ગઈ!")
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
    
    # પાસવર્ડ ચેક
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
                            if os.path.exists(f_item["file_path"]):
                                matched.append((f_item["file_path"], f_item["photo_name"], f_item.get("person_label", "Guest")))
                                seen_photos.add(f_item["photo_name"])
                                
                if matched:
                    st.success(f"🎉 તમારા {len(matched)} ફોટા મળ્યા!")
                    cols = st.columns(3)
                    for idx, (p_path, p_name, p_lbl) in enumerate(matched):
                        with cols[idx % 3]:
                            st.image(p_path, caption=f"{p_name} ({p_lbl})", use_container_width=True)
                            with open(p_path, "rb") as f:
                                st.download_button("⬇️ ડાઉનલોડ", f, file_name=p_name, key=f"dl_{idx}")
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
            # તમારી ખરી લિંક
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