import streamlit as st
import os
import base64
import hashlib
import hmac  # constant-time comparison
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.decomposition import PCA

# ---------- Optional warning control (warnings are allowed for debugging) ----------

# ---------- Supabase import with fallback ----------
try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    st.warning("⚠️ Supabase not installed. Authentication will not work.")

# ---------- PyCaret imports (optional) ----------
try:
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare, predict_model as clf_predict
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, predict_model as reg_predict, get_config as reg_get_config
    from pycaret.classification import get_config as clf_get_config
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    st.warning("⚠️ PyCaret not installed. Classification/Regression will use scikit-learn fallback.")

# ---------- Scipy for outlier detection (not used after cleaning removal, but kept for completeness) ----------
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------- PDF generator ----------
def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def text_to_simple_pdf_bytes(text: str, title: str = "ML Model Report") -> bytes:
    page_w, page_h = 612, 792
    margin_x, margin_y = 54, 54
    font_size = 10
    leading = 14
    max_lines = int((page_h - 2 * margin_y) / leading)

    lines = (text or "").splitlines() or ["(empty report)"]
    pages = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]

    objects: List[bytes] = []

    def add_obj(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog_obj_num = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_obj(b"<< /Type /Pages /Kids [] /Count 0 >>")

    page_obj_nums: List[int] = []
    for page_lines in pages:
        y0 = page_h - margin_y
        text_ops = [b"BT", b"/F1 %d Tf" % font_size, b"1 0 0 1 %d %d Tm" % (margin_x, y0)]
        for i, line in enumerate(page_lines):
            if i > 0:
                text_ops.append(b"0 -%d Td" % leading)
            text_ops.append(b"(%s) Tj" % _pdf_escape(line).encode("utf-8"))
        text_ops.append(b"ET")
        stream = b"\n".join(text_ops) + b"\n"

        content_obj_num = add_obj(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream"
        )

        page_obj = b"".join(
            [
                b"<< /Type /Page /Parent 2 0 R ",
                (b"/MediaBox [0 0 %d %d] " % (page_w, page_h)),
                b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> ",
                (b"/Contents %d 0 R >>" % content_obj_num),
            ]
        )
        page_obj_nums.append(add_obj(page_obj))

    kids = b" ".join([b"%d 0 R" % n for n in page_obj_nums])
    objects[1] = b"<< /Type /Pages /Kids [ %s ] /Count %d >>" % (kids, len(page_obj_nums))

    info_obj_num = add_obj(b"<< /Title (%s) >>" % _pdf_escape(title).encode("utf-8"))

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = b""
    offsets = [0]
    cur = len(header)

    for i, obj in enumerate(objects, start=1):
        offsets.append(cur)
        obj_bytes = b"%d 0 obj\n%s\nendobj\n" % (i, obj)
        body += obj_bytes
        cur += len(obj_bytes)

    xref_start = len(header) + len(body)
    xref = [b"xref\n0 %d\n" % (len(objects) + 1), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(b"%010d 00000 n \n" % off)
    xref_bytes = b"".join(xref)

    trailer = (
        b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%EOF\n"
        % (len(objects) + 1, catalog_obj_num, info_obj_num, xref_start)
    )

    return header + body + xref_bytes + trailer

# ---------- Background image helper (fixed MIME type) ----------
def get_base64_of_file(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_bg_image_local(image_path):
    """Set background image with correct MIME type based on file extension."""
    bin_str = get_base64_of_file(image_path)
    if bin_str:
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:{mime};base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        fallback_bg = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """
        st.markdown(fallback_bg, unsafe_allow_html=True)

# ---------- Password hashing (with constant-time verification) ----------
def hash_password(password: str, iterations: int = 100_000) -> str:
    salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return (
        f"pbkdf2_sha256${iterations}$"
        f"{base64.b64encode(salt).decode('utf-8')}$"
        f"{base64.b64encode(pwd_hash).decode('utf-8')}"
    )

def verify_password(plain_password: str, stored_password: str) -> bool:
    if not stored_password:
        return False
    if stored_password.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_str, salt_b64, hash_b64 = stored_password.split("$", 3)
            iterations = int(iterations_str)
            salt = base64.b64decode(salt_b64.encode("utf-8"))
            expected_hash = base64.b64decode(hash_b64.encode("utf-8"))
            candidate_hash = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(candidate_hash, expected_hash)
        except Exception:
            return False
    # Fallback for plaintext (not recommended, kept for backward compatibility)
    return hmac.compare_digest(plain_password, stored_password)

# ---------- Supabase client (only if available) ----------
if "supabase" not in st.session_state:
    if SUPABASE_AVAILABLE:
        try:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            st.session_state.supabase = create_client(url, key)
        except Exception as e:
            st.error(f"Supabase connection failed: {e}")
            st.session_state.supabase = None
    else:
        st.session_state.supabase = None

def register_user(email, password, name):
    if st.session_state.supabase is None:
        return False, "Supabase not connected"
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) > 0:
            return False, "Email already registered."
        data = {"email": email, "name": name, "password": hash_password(password)}
        st.session_state.supabase.table("users").insert(data).execute()
        return True, "Registration successful. Please log in."
    except Exception as e:
        return False, f"Registration failed: {e}"

def authenticate_user(email, password):
    if st.session_state.supabase is None:
        return False, None, None
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) == 0:
            return False, None, None
        user = response.data[0]
        if verify_password(password, user.get("password", "")):
            return True, user["name"], user["email"]
        return False, None, None
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False, None, None

# ---------- Centralized state reset for ML results ----------
def reset_ml_state():
    """Reset all ML-related session state keys to their defaults."""
    defaults = {
        "data": None,
        "target_column": None,
        "problem_type": None,
        "model": None,
        "predictions": None,
        "test_labels": None,
        "training_complete": False,
        "cleaned_data": None,
        "feature_names": None,
        "training_done": False,
        "cluster_labels": None,
        "cluster_metrics": None,
        "clustering_model": None,
        "clustering_scaler": None,
        "clustering_X_scaled": None,   # Added to store scaled data for clustering
    }
    for k, v in defaults.items():
        st.session_state[k] = v

# ---------- Page navigation ----------
if "page" not in st.session_state:
    st.session_state.page = "front"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

def go_to(page: str):
    if st.session_state.page != page:
        st.session_state.page = page
        st.rerun()

st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CSS styling ----------
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; padding: 1rem; margin-bottom: 2rem; }
    .sub-header { font-size: 1.5rem; color: #3949AB; margin-top: 1.5rem; margin-bottom: 1rem; }
    .card {
        background-color: rgba(255,255,255,0.85);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #2196F3, #9C27B0) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.2rem !important;
        border-radius: 50px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4) !important;
    }
    section[data-testid="stSidebar"] {
        background: #ffffe0 !important;
    }
    section[data-testid="stSidebar"] .st-emotion-cache-1wrcr25, 
    section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state initialisation (already done by reset_ml_state) ----------
if "data" not in st.session_state:
    reset_ml_state()

# ---------- Safe PyCaret setup (non-recursive, using inspect.signature) ----------
import inspect
def _pycaret_setup_safe(setup_fn, **kwargs):
    """
    Call PyCaret setup() with only the arguments it accepts.
    Uses inspect.signature to filter out unexpected keyword arguments.
    """
    sig = inspect.signature(setup_fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return setup_fn(**filtered)

# ---------- Task suitability detection ----------
def is_classification_possible(df) -> Tuple[bool, List[str]]:
    candidates = []
    for col in df.columns:
        dtype = df[col].dtype
        unique_vals = df[col].nunique(dropna=False)
        if dtype in ['object', 'category']:
            candidates.append(col)
        elif np.issubdtype(dtype, np.number) and unique_vals < 20:
            candidates.append(col)
    return len(candidates) > 0, candidates

def is_regression_possible(df) -> Tuple[bool, List[str]]:
    candidates = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number) and df[col].nunique(dropna=False) >= 20:
            candidates.append(col)
    return len(candidates) > 0, candidates

def is_clustering_possible(df, min_rows=10, min_numeric_features=2) -> Tuple[bool, str]:
    if len(df) < min_rows:
        return False, f"Data rows insufficient: need at least {min_rows} (currently {len(df)})"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < min_numeric_features:
        return False, f"Numeric features insufficient: need at least {min_numeric_features} (currently {len(numeric_cols)})"
    constant_cols = []
    for col in numeric_cols:
        if df[col].var() == 0:
            constant_cols.append(col)
    if constant_cols:
        return False, f"Constant numeric features: {', '.join(constant_cols[:3])}"
    return True, "Suitable for clustering"

# ---------- AutoML for Clustering (returns scaled data for visualization) ----------
def auto_clustering(df, max_clusters=10, skip_hierarchical=False, skip_birch=False, skip_dbscan=False):
    """
    Automatically find best clustering algorithm and number of clusters.
    
    Parameters:
    - df: DataFrame with numeric features only
    - max_clusters: maximum number of clusters to try (for KMeans, Hierarchical, BIRCH)
    - skip_hierarchical: if True, skip AgglomerativeClustering
    - skip_birch: if True, skip BIRCH
    - skip_dbscan: if True, skip DBSCAN
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    
    best_score = -1
    best_model = None
    best_labels = None
    best_name = None
    
    # 1. KMeans (always run)
    for k in range(2, min(max_clusters, len(df)-1)+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_model = km
                best_labels = labels
                best_name = f"KMeans (k={k})"
    
    # 2. Hierarchical (optional)
    if not skip_hierarchical:
        for linkage in ['ward', 'complete', 'average']:
            try:
                for k in range(2, min(max_clusters, len(df)-1)+1):
                    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                    labels = hc.fit_predict(X_scaled)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_scaled, labels)
                        if score > best_score:
                            best_score = score
                            best_model = hc
                            best_labels = labels
                            best_name = f"Agglomerative (k={k}, linkage={linkage})"
            except Exception:
                continue
    
    # 3. BIRCH (optional)
    if not skip_birch:
        for k in range(2, min(max_clusters, len(df)-1)+1):
            try:
                birch = Birch(n_clusters=k)
                labels = birch.fit_predict(X_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_model = birch
                        best_labels = labels
                        best_name = f"BIRCH (k={k})"
            except Exception:
                continue
    
    # 4. DBSCAN (optional)
    if not skip_dbscan:
        eps_range = np.linspace(0.1, 1.5, 10)
        for eps in eps_range:
            try:
                db = DBSCAN(eps=eps, min_samples=5)
                labels = db.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters >= 2:
                    mask = labels != -1
                    if mask.sum() >= 2:
                        score = silhouette_score(X_scaled[mask], labels[mask])
                        if score > best_score:
                            best_score = score
                            best_model = db
                            best_labels = labels
                            best_name = f"DBSCAN (eps={eps:.2f})"
            except Exception:
                continue
    
    # Fallback if nothing found
    if best_model is None:
        best_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_labels = best_model.fit_predict(X_scaled)
        best_name = "KMeans (k=2, fallback)"
        best_score = silhouette_score(X_scaled, best_labels) if len(set(best_labels)) > 1 else 0
    
    metrics = {
        "silhouette": best_score,
        "calinski_harabasz": calinski_harabasz_score(X_scaled, best_labels) if len(set(best_labels)) > 1 else 0,
        "davies_bouldin": davies_bouldin_score(X_scaled, best_labels) if len(set(best_labels)) > 1 else 0,
        "num_clusters": len(set(best_labels)),
        "algorithm": best_name,
        "cluster_sizes": pd.Series(best_labels).value_counts().to_dict()
    }
    return best_model, best_labels, best_name, best_score, metrics, scaler, X_scaled

# ---------- Fallback training ----------
def train_fallback_model(df, target_col, problem_type):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if problem_type == "Classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    st.session_state.feature_names = X.columns.tolist()
    return model, preds, y_test.values

# ---------- Page functions ----------
def front_page():
    set_bg_image_local("FrontPage.jpg")
    st.markdown("""
    <style>.stApp { color: white !important; } .stApp * { color: white !important; } div.stButton > button { color: white !important; }</style>
    <style>
    .right-panel {
        background-color: rgba(0, 0, 0, 0.70);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        animation: fadeIn 1s ease-in-out;
        color: white;
        height: 100%;
    }
    .right-panel h1 { text-shadow: 2px 2px 4px rgba(0,0,0,0.5); font-size: 3rem; margin-bottom: 1rem; }
    .right-panel p { text-shadow: 1px 1px 2px rgba(0,0,0,0.5); font-size: 1.2rem; opacity: 0.9; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .video-container {
        display: flex;
        align-items: center;
        justify-content: center;
        max-height: 400px;
        height: auto;
        margin: auto;
    }
    .video-container video {
        width: 100%;
        height: auto;
        max-height: 400px;
        object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        video_path = "animation.mp4"
        if os.path.exists(video_path):
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode()
            video_html = f'''
            <div class="video-container">
                <video autoplay loop muted playsinline>
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                </video>
            </div>
            '''
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 2rem; text-align: center;"><span style="font-size: 3rem;">📹</span><p style="color: white;">Video not found. Please add animation.mp4</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="right-panel"><h1>Welcome to<br>No-Code ML Platform</h1><p>Accessible Machine Learning without code.</p></div>', unsafe_allow_html=True)
        if st.button("Get Started", key="get_started", use_container_width=True):
            go_to("login")

def login_page():
    set_bg_image_local("login.jpg")
    st.markdown("""
    <style>
    .stApp { display: flex; align-items: center; justify-content: center; }
    .stTabs [data-baseweb="tab-list"] button { color: rgba(255,255,255,0.8); font-size: 1.1rem; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { color: white; border-bottom-color: #2196F3; }
    .stTextInput input { color: black !important; background-color: rgba(255,255,255,0.1) !important; border: 1px solid rgba(255,255,255,0.3) !important; border-radius: 5px; }
    .stTextInput label { color: black !important; }
    .back-button-container { text-align: center; margin-top: 1.5rem; }
    .back-button-container button { background: transparent !important; color: rgba(255,255,255,0.9) !important; border: 1px solid rgba(255,255,255,0.3) !important; padding: 0.5rem 1.5rem !important; border-radius: 50px !important; font-size: 1rem !important; transition: all 0.2s ease !important; width: auto !important; display: inline-block !important; box-shadow: none !important; }
    .back-button-container button:hover { background: rgba(255,255,255,0.1) !important; border-color: rgba(255,255,255,0.6) !important; transform: scale(1.02) !important; }
    </style>
    """, unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown("<h2 style='color: white; text-align: center; margin-bottom: 1.5rem;'>Login / Register</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    success, name, email_ret = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_name = name
                        st.session_state.user_email = email_ret
                        go_to("dashboard")
                    else:
                        st.error("Invalid email or password")
        with tab2:
            with st.form("register_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register")
                if submitted:
                    if password != confirm:
                        st.error("Passwords do not match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, msg = register_user(email, password, name)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
        st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
        if st.button("← Back to Home", key="back_home"):
            go_to("front")
        st.markdown('</div>')
        st.markdown('</div>')

def upload_page():
    st.markdown('<h2 class="sub-header">📁 Upload Your Dataset</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        last_error = None
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                st.error(f"Error reading CSV with encoding {enc}: {e}")
                last_error = e
                continue
        if df is None:
            st.error(f"Could not read CSV file. Last error: {last_error}")
            return
        # New data loaded → reset ML state (preserve nothing)
        reset_ml_state()
        st.session_state.data = df
        st.success(f"✔️ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        class_possible, class_candidates = is_classification_possible(df)
        reg_possible, reg_candidates = is_regression_possible(df)
        clust_possible, clust_msg = is_clustering_possible(df)
        
        available_tasks = []
        if class_possible:
            available_tasks.append("Classification")
        if reg_possible:
            available_tasks.append("Regression")
        if clust_possible:
            available_tasks.append("Clustering")
        
        st.markdown("### 🎯 Auto‑detected Task Suggestion")
        if available_tasks:
            bullet_list = "\n".join([f"- ✅ **{task}**" for task in available_tasks])
            st.info(f"**This dataset is suitable for:**\n{bullet_list}")
        else:
            st.error("❌ No machine learning task is possible with this dataset. Please upload another CSV.")
            return
        
        st.markdown("### 🔍 Detected Target Candidates")
        col1a, col2a = st.columns(2)
        with col1a:
            st.markdown("**Classification Candidates**")
            if class_candidates:
                st.write(", ".join(class_candidates))
            else:
                st.write("None detected")
        with col2a:
            st.markdown("**Regression Candidates**")
            if reg_candidates:
                st.write(", ".join(reg_candidates))
            else:
                st.write("None detected")
        st.markdown("---")
        
        st.markdown("### 📌 Define Problem Type")
        problem_type = st.selectbox("Select problem type:", available_tasks)
        
        if problem_type == "Clustering":
            st.info("Clustering is unsupervised – no target column required. The system will automatically select the best algorithm and number of clusters.")
            if st.button("Set Clustering Task", type="primary", key="set_clustering"):
                reset_ml_state()  # Clear previous ML state
                st.session_state.data = df  # restore data
                st.session_state.target_column = None
                st.session_state.problem_type = "Clustering"
                st.success("✅ Clustering task selected. Proceed to Model Training for AutoML.")
        else:
            if problem_type == "Classification":
                candidates = class_candidates
            else:
                candidates = reg_candidates
            if not candidates:
                st.error(f"❌ No suitable target column found for {problem_type}. Please check your data.")
                return
            target_col = st.selectbox(f"Select target column for {problem_type}:", candidates)
            if st.button("Set Target", type="primary", key="set_target"):
                if problem_type == "Classification" and df[target_col].nunique() > 50:
                    st.warning(f"⚠️ Target column '{target_col}' has {df[target_col].nunique()} unique values. Classification may be difficult.")
                elif problem_type == "Regression" and not np.issubdtype(df[target_col].dtype, np.number):
                    st.error(f"❌ Target column '{target_col}' is not numeric. Regression requires a numeric target.")
                    return
                reset_ml_state()  # Clear previous ML state
                st.session_state.data = df
                st.session_state.target_column = target_col
                st.session_state.problem_type = problem_type
                st.success(f"✅ Target set: {target_col} ({problem_type})")
        
        st.markdown("---")
        st.markdown("### Data Preview")
        st.dataframe(df.head(), width='stretch')
        
        with st.expander("📊 Basic Data Statistics"):
            st.write("**Shape:**", df.shape)
            col_types = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_types, width='stretch')
    else:
        st.info("📂 No data loaded yet. Please upload a CSV file.")

def eda_page():
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data and set target column first.")
        return
    st.markdown('<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    df = st.session_state.data
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Rows", len(df))
    with info_col2:
        st.metric("Columns", len(df.columns))
    with info_col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    with info_col4:
        memory = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory (MB)", f"{memory:.2f}")
    st.markdown("### 🔍 Data Types")
    dtype_counts = df.dtypes.value_counts()
    dtype_df = pd.DataFrame({'Data Type': dtype_counts.index.astype(str), 'Count': dtype_counts.values})
    fig = px.pie(dtype_df, values='Count', names='Data Type', title="Distribution of Data Types")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### ⚠️ Missing Values Analysis")
    missing_series = df.isnull().sum()
    missing_df = pd.DataFrame({'Column': missing_series.index, 'Missing_Count': missing_series.values, 'Missing_Percentage': (missing_series.values / len(df)) * 100}).sort_values('Missing_Percentage', ascending=False)
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    if len(missing_df) > 0:
        fig = px.bar(missing_df, x='Column', y='Missing_Percentage', title="Missing Values by Column (%)", color='Missing_Percentage')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(missing_df, width='stretch')
    else:
        st.success("✅ No missing values found!")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        st.markdown("### 📊 Numerical Columns Analysis")
        selected_num_col = st.selectbox("Select numerical column:", numerical_cols)
        if selected_num_col:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}", nbins=50)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
                st.plotly_chart(fig, use_container_width=True)
            col_stats = df[selected_num_col].describe()
            st.dataframe(col_stats, width='stretch')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        st.markdown("### 📊 Categorical Columns Analysis")
        selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
        if selected_cat_col:
            value_counts = df[selected_cat_col].value_counts().head(20)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Top Categories in {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(names=value_counts.index, values=value_counts.values, title=f"Distribution of {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)
    if len(numerical_cols) > 1:
        st.markdown("### 🔗 Correlation Matrix")
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix, labels=dict(color="Correlation"), x=corr_matrix.columns, y=corr_matrix.columns, title="Correlation Heatmap")
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)
    if st.session_state.target_column and st.session_state.target_column in df.columns:
        st.markdown(f"### 🎯 Analysis of Target: {st.session_state.target_column}")
        target_col = st.session_state.target_column
        if df[target_col].dtype in ['int64', 'float64']:
            fig = px.histogram(df, x=target_col, title=f"Distribution of Target ({target_col})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            value_counts = df[target_col].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(x=value_counts.index, y=value_counts.values, title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(names=value_counts.index, values=value_counts.values, title="Class Proportions")
                st.plotly_chart(fig, use_container_width=True)

def clustering_training_page():
    """Clustering training page with user-selectable search scope."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return
    st.markdown('<h2 class="sub-header">🎯 Automated Clustering (Unsupervised AutoML)</h2>', unsafe_allow_html=True)
    df = st.session_state.data.copy()
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.error("Clustering requires at least 2 numeric features. Current numeric features insufficient.")
        return
    
    # Speed mode selection
    cluster_mode = st.radio(
        "⚡ Clustering Search Mode",
        options=["Fast (KMeans only, k≤5)", "Standard (KMeans + Hierarchical)", "Full (try all algorithms)"],
        index=1,
        help="Fast: a few seconds. Standard: includes KMeans and Agglomerative. Full: also BIRCH and DBSCAN (slower)."
    )
    
    st.markdown(f"""
    <div class="card">
    <h4>AutoClustering Configuration</h4>
    <ul><li><strong>Dataset Shape:</strong> {df.shape}</li>
    <li><strong>Numerical features used:</strong> {numeric_df.shape[1]}</li>
    <li><strong>Unsupervised – no target column</strong></li>
    <li><strong>Search Mode:</strong> {cluster_mode}</li></ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Run AutoML Clustering", type="primary"):
        with st.spinner("Automatically searching for best clustering algorithm and parameters..."):
            try:
                # Set flags based on mode
                if cluster_mode == "Fast (KMeans only, k≤5)":
                    max_clusters = 5
                    skip_hierarchical = True
                    skip_birch = True
                    skip_dbscan = True
                elif cluster_mode == "Standard (KMeans + Hierarchical)":
                    max_clusters = 8
                    skip_hierarchical = False
                    skip_birch = True
                    skip_dbscan = True
                else:  # Full mode
                    max_clusters = 10
                    skip_hierarchical = False
                    skip_birch = False
                    skip_dbscan = False
                
                model, labels, algo_name, score, metrics, scaler, X_scaled = auto_clustering(
                    numeric_df, 
                    max_clusters=max_clusters,
                    skip_hierarchical=skip_hierarchical,
                    skip_birch=skip_birch,
                    skip_dbscan=skip_dbscan
                )
                
                st.session_state.cluster_labels = labels
                st.session_state.clustering_model = model
                st.session_state.training_complete = True
                st.session_state.training_done = True
                st.session_state.problem_type = "Clustering"
                st.session_state.cluster_metrics = {
                    "algorithm": algo_name,
                    "num_clusters": metrics["num_clusters"],
                    "silhouette_score": metrics["silhouette"],
                    "calinski_harabasz": metrics["calinski_harabasz"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "cluster_sizes": metrics["cluster_sizes"]
                }
                st.session_state.clustering_scaler = scaler
                st.session_state.clustering_X_scaled = X_scaled   # Save scaled data for evaluation
                st.success(f"🎉 AutoML completed! Best algorithm: {algo_name} (Silhouette = {score:.4f})")
                
                with st.expander("📊 Clustering Results", expanded=True):
                    st.markdown("#### Best Model")
                    st.code(f"Algorithm: {algo_name}\nNumber of clusters: {metrics['num_clusters']}\nSilhouette Score: {score:.4f}")
                    st.markdown("#### Cluster Sizes")
                    sizes_df = pd.DataFrame(list(metrics["cluster_sizes"].items()), columns=["Cluster", "Count"])
                    fig = px.bar(sizes_df, x="Cluster", y="Count", title="Number of points per cluster")
                    st.plotly_chart(fig, use_container_width=True)
                    if X_scaled.shape[1] >= 2:
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(X_scaled)
                        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
                        pca_df['Cluster'] = labels.astype(str)
                        fig2 = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="PCA Projection of Clusters (scaled data)")
                        st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"AutoML clustering failed: {e}")
                st.exception(e)

def training_page():
    """Model training with full auto fallback – never shows an error."""
    if st.session_state.problem_type == "Clustering":
        clustering_training_page()
        return
    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload data and set target column first.")
        return
    st.markdown('<h2 class="sub-header">📐 Automated Model Training (AutoML)</h2>', unsafe_allow_html=True)
    if not PYCARET_AVAILABLE:
        st.info("PyCaret not installed. Using scikit-learn fallback (RandomForest).")
    
    df = st.session_state.data.copy()
    target_col = st.session_state.target_column
    problem_type = st.session_state.problem_type
    
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found.")
        return
    if df[target_col].isnull().sum() > 0:
        st.error(f"Target column '{target_col}' contains missing values. Please clean first.")
        return
    if problem_type == "Regression" and not pd.api.types.is_numeric_dtype(df[target_col]):
        st.error(f"Target column '{target_col}' must be numeric for regression.")
        return
    
    # Speed mode selection (still offered, but will auto fallback)
    train_mode = st.radio(
        "⚡ Training Speed Mode",
        options=["Fast (few seconds)", "Standard (1-2 min)", "Full (slower, best)"],
        index=1,
        help="Fast: RandomForest only, 3‑fold. Standard: 5 models, 5‑fold. Full: all models, 10‑fold. If PyCaret fails, auto fallback to RandomForest."
    )
    
    st.markdown(f"""
    <div class="card">
    <h4>AutoML Configuration</h4>
    <ul><li><strong>Problem Type:</strong> {problem_type}</li>
    <li><strong>Target Column:</strong> {target_col}</li>
    <li><strong>Dataset Shape:</strong> {df.shape}</li>
    <li><strong>Speed Mode:</strong> {train_mode}</li></ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Run AutoML Training", type="primary"):
        with st.spinner("Training... (auto fallback enabled)"):
            try:
                trained = False
                model = None
                preds = None
                y_true = None
                
                # ---- Try PyCaret first if available ----
                if PYCARET_AVAILABLE:
                    try:
                        # Set parameters based on mode
                        if train_mode == "Fast (few seconds)":
                            fold = 3
                            include_models = ['rf']   # RandomForest only
                        elif train_mode == "Standard (1-2 min)":
                            fold = 5
                            if problem_type == "Classification":
                                include_models = ['lr', 'rf', 'xgboost', 'lightgbm', 'dt']
                            else:
                                include_models = ['lr', 'rf', 'lightgbm', 'dt', 'et']
                        else:  # Full mode
                            fold = 10
                            include_models = None  # all models
                        
                        sort_metric = 'Accuracy' if problem_type == 'Classification' else 'R2'
                        
                        # Setup with auto-preprocess (let PyCaret handle everything)
                        # Note: fold is NOT passed to setup; it's only for compare_models
                        setup_args = {
                            "data": df,
                            "target": target_col,
                            "train_size": 0.8,
                            "session_id": 42,
                            "verbose": False,
                            "log_experiment": False,
                            "n_jobs": -1,
                            "html": False,
                            "preprocess": True,   # Auto preprocessing: impute, encode, scale
                        }
                        
                        if problem_type == "Classification":
                            _pycaret_setup_safe(clf_setup, **setup_args)
                            best_model = clf_compare(verbose=False, sort=sort_metric, include=include_models, n_select=1, fold=fold)
                            # Handle case where compare_models returns None or empty list
                            if best_model is None or (isinstance(best_model, list) and len(best_model) == 0):
                                raise ValueError("compare_models returned no model")
                            if isinstance(best_model, list):
                                best_model = best_model[0]
                            pred_df = clf_predict(best_model)
                            preds = pred_df['prediction_label'].values
                            y_true = pred_df[target_col].values
                            st.session_state.feature_names = clf_get_config('X_train').columns.tolist()
                        else:  # Regression
                            _pycaret_setup_safe(reg_setup, **setup_args)
                            best_model = reg_compare(verbose=False, sort=sort_metric, include=include_models, n_select=1, fold=fold)
                            if best_model is None or (isinstance(best_model, list) and len(best_model) == 0):
                                raise ValueError("compare_models returned no model")
                            if isinstance(best_model, list):
                                best_model = best_model[0]
                            pred_df = reg_predict(best_model)
                            preds = pred_df['prediction_label'].values
                            y_true = pred_df[target_col].values
                            st.session_state.feature_names = reg_get_config('X_train').columns.tolist()
                        
                        model = best_model
                        trained = True
                        st.success("✅ AutoML training completed with PyCaret!")
                    except Exception as pycaret_error:
                        st.warning(f"⚠️ PyCaret training failed: {pycaret_error}. Falling back to scikit-learn (RandomForest).")
                        # Fall through to fallback
                        trained = False
                
                # ---- Fallback to scikit-learn if PyCaret not available or failed ----
                if not trained:
                    model, preds, y_true = train_fallback_model(df, target_col, problem_type)
                    st.info("ℹ️ Used scikit-learn RandomForest (fallback).")
                
                # Store results
                st.session_state.model = model
                st.session_state.predictions = preds
                st.session_state.test_labels = y_true
                st.session_state.training_complete = True
                st.success("Training completed successfully!")
            except Exception as e:
                st.error(f"Training completely failed: {e}")
                st.exception(e)

def evaluation_page():
    if not st.session_state.training_complete:
        st.warning("⚠️ No trained model found. Please go to 'Model Training' and train a model first.")
        return
    
    if st.session_state.problem_type == "Clustering":
        st.markdown('<h2 class="sub-header">📈 Clustering Performance Evaluation</h2>', unsafe_allow_html=True)
        if st.session_state.cluster_labels is None:
            st.error("No clustering labels found.")
            return
        
        labels = st.session_state.cluster_labels
        df = st.session_state.data.copy()
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            st.info("Not enough numeric columns for clustering metrics.")
        else:
            # Use saved scaled data if available, otherwise recompute
            X_scaled = st.session_state.get("clustering_X_scaled")
            if X_scaled is not None:
                # Ensure shape matches number of rows
                if len(X_scaled) != len(numeric_df):
                    st.warning("Saved scaled data length mismatch. Recomputing...")
                    scaler = st.session_state.get("clustering_scaler")
                    if scaler is not None:
                        X_scaled = scaler.transform(numeric_df)
                    else:
                        X_scaled = numeric_df.values
            else:
                scaler = st.session_state.get("clustering_scaler")
                if scaler is not None:
                    X_scaled = scaler.transform(numeric_df)
                else:
                    st.warning("Scaler not found, using raw data (may be inaccurate).")
                    X_scaled = numeric_df.values
            
            sil = silhouette_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette Score", f"{sil:.4f}")
            col2.metric("Calinski-Harabasz", f"{ch:.2f}")
            col3.metric("Davies-Bouldin", f"{db:.4f}")
            
            if st.session_state.cluster_metrics:
                with st.expander("📋 Training-time Metrics (AutoML selection)"):
                    st.write(f"**Algorithm chosen:** {st.session_state.cluster_metrics['algorithm']}")
                    st.write(f"**Silhouette (training):** {st.session_state.cluster_metrics['silhouette_score']:.4f}")
                    st.write(f"**Calinski-Harabasz:** {st.session_state.cluster_metrics['calinski_harabasz']:.2f}")
                    st.write(f"**Davies-Bouldin:** {st.session_state.cluster_metrics['davies_bouldin']:.4f}")
        
        st.markdown("### Cluster Distribution")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values, labels={'x':'Cluster','y':'Count'})
        st.plotly_chart(fig, use_container_width=True)
        
        if numeric_df.shape[1] >= 2:
            # Use saved scaled data for PCA if available
            X_scaled = st.session_state.get("clustering_X_scaled")
            if X_scaled is None:
                scaler = st.session_state.get("clustering_scaler")
                if scaler is not None:
                    X_scaled = scaler.transform(numeric_df)
                else:
                    X_scaled = numeric_df.values
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pca_result, columns=['PC1','PC2'])
            pca_df['Cluster'] = labels.astype(str)
            fig2 = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="PCA Projection")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### Cluster Assignments (first 100 rows)")
        df_with_cluster = df.copy()
        df_with_cluster['Cluster'] = labels
        st.dataframe(df_with_cluster[['Cluster'] + [c for c in df_with_cluster.columns if c != 'Cluster']].head(100), width='stretch')
        return
    
    # Classification / Regression evaluation
    st.markdown('<h2 class="sub-header">📈 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    preds = st.session_state.predictions
    y_true = st.session_state.test_labels
    problem = st.session_state.problem_type
    
    if problem == "Classification":
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, average='weighted', zero_division=0)
        rec = recall_score(y_true, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_true, preds, average='weighted', zero_division=0)
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.4f}")
        col1.metric("Precision", f"{prec:.4f}")
        col2.metric("Recall", f"{rec:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")
        cm = confusion_matrix(y_true, preds)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        report = classification_report(y_true, preds, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), width='stretch')
    else:
        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2:.4f}")
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        residuals = y_true - preds
        fig1 = px.scatter(x=preds, y=residuals, labels={'x':'Predicted','y':'Residuals'}, title="Residuals vs Predicted")
        fig1.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.scatter(x=y_true, y=preds, labels={'x':'Actual','y':'Predicted'}, title="Actual vs Predicted")
        fig2.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], mode='lines', name='Ideal', line=dict(dash='dash', color='red')))
        st.plotly_chart(fig2, use_container_width=True)

def export_page():
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first to export results.")
        return
    st.markdown('<h2 class="sub-header">💾 Export Model and Results</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Model Information")
        if st.button("Show Model Details"):
            if st.session_state.problem_type == "Clustering":
                st.write("Clustering Model:", st.session_state.clustering_model)
                st.write("Algorithm:", st.session_state.cluster_metrics.get("algorithm", "N/A") if st.session_state.cluster_metrics else "N/A")
            else:
                st.write("Best Model:", st.session_state.model)
    with col2:
        st.markdown("#### 📥 Download Predictions / Cluster Assignments")
        if st.session_state.problem_type == "Clustering" and st.session_state.cluster_labels is not None:
            results_df = st.session_state.data.copy()
            results_df["Cluster"] = st.session_state.cluster_labels
            st.download_button("Download Cluster Assignments (CSV)", results_df.to_csv(index=False), "cluster_assignments.csv")
        elif st.session_state.predictions is not None:
            results_df = pd.DataFrame({"Actual": st.session_state.test_labels, "Predicted": st.session_state.predictions})
            st.download_button("Download Predictions (CSV)", results_df.to_csv(index=False), "predictions.csv")
        else:
            st.warning("No predictions or cluster labels available.")
    st.markdown("#### 📄 Model Report")
    if st.button("Generate Model Report"):
        dataset_shape = st.session_state.data.shape if st.session_state.data is not None else "N/A"
        if st.session_state.problem_type == "Clustering":
            report = f"""# Clustering Model Report (AutoML)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Best Algorithm: {st.session_state.cluster_metrics.get('algorithm', 'N/A') if st.session_state.cluster_metrics else 'N/A'}
Dataset shape: {dataset_shape}
Number of clusters: {st.session_state.cluster_metrics.get('num_clusters', 'N/A') if st.session_state.cluster_metrics else 'N/A'}
Silhouette Score: {st.session_state.cluster_metrics.get('silhouette_score', 'N/A') if st.session_state.cluster_metrics else 'N/A'}
Calinski-Harabasz: {st.session_state.cluster_metrics.get('calinski_harabasz', 'N/A') if st.session_state.cluster_metrics else 'N/A'}
Davies-Bouldin: {st.session_state.cluster_metrics.get('davies_bouldin', 'N/A') if st.session_state.cluster_metrics else 'N/A'}
"""
        else:
            report = f"""# Machine Learning Model Report (AutoML)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Problem Type: {st.session_state.problem_type}
Target Column: {st.session_state.target_column}
Dataset shape: {dataset_shape}
Model: {st.session_state.model}
Training completed: {st.session_state.training_complete}
"""
        st.code(report, language='markdown')
        pdf_bytes = text_to_simple_pdf_bytes(report, title="ML Model Report")
        st.download_button("📥 Download Report (PDF)", pdf_bytes, "ml_model_report.pdf")
    
    st.markdown("### 📋 Session Information")
    session_info = {
        "Data Loaded": str(st.session_state.data is not None),
        "Problem Type": str(st.session_state.problem_type) if st.session_state.problem_type else "N/A",
        "Target Column": str(st.session_state.target_column) if st.session_state.target_column else "N/A",
        "Model Trained": str(st.session_state.training_complete),
        "Predictions/Clusters Available": str((st.session_state.predictions is not None) or (st.session_state.cluster_labels is not None))
    }
    st.dataframe(pd.DataFrame.from_dict(session_info, orient='index', columns=['Status']), width='stretch')
    
    if st.button("🔄 Start Over", type="secondary"):
        reset_ml_state()
        go_to("data_upload")

def account_page():
    st.markdown('<h2 class="sub-header">👤 Account Settings</h2>', unsafe_allow_html=True)
    st.markdown("### Your Profile")
    st.write(f"**Name:** {st.session_state.user_name}")
    st.write(f"**Email:** {st.session_state.user_email}")
    st.markdown("---")
    st.markdown("### Change Password")
    with st.form("change_password_form"):
        current = st.text_input("Current Password", type="password")
        new = st.text_input("New Password", type="password")
        confirm = st.text_input("Confirm New Password", type="password")
        if st.form_submit_button("Update Password"):
            if not current or not new or not confirm:
                st.error("Fill all fields.")
            elif new != confirm:
                st.error("Passwords do not match.")
            elif len(new) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                if st.session_state.supabase is None:
                    st.error("Supabase not connected.")
                else:
                    try:
                        resp = st.session_state.supabase.table("users").select("*").eq("email", st.session_state.user_email).execute()
                        if len(resp.data) == 0:
                            st.error("User not found.")
                        else:
                            user = resp.data[0]
                            if verify_password(current, user.get("password", "")):
                                new_hash = hash_password(new)
                                st.session_state.supabase.table("users").update({"password": new_hash}).eq("email", st.session_state.user_email).execute()
                                st.success("Password updated!")
                            else:
                                st.error("Current password is incorrect.")
                    except Exception as e:
                        st.error(f"Failed: {e}")

def dashboard_page():
    set_bg_image_local("purple.png")
    st.markdown(f"<h1 style='color: black;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)
    # Removed "data_cleaning" from workflow
    workflow_pages = ["data_upload", "eda", "model_training", "model_evaluation", "export_results", "account"]
    page_display = {
        "data_upload": "📁 Data Upload",
        "eda": "🔍 Exploratory Data Analysis",
        "model_training": "📐 AutoML Training",
        "model_evaluation": "📈 Model Evaluation",
        "export_results": "💾 Export Results",
        "account": "👤 Account Settings"
    }
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
        st.markdown("---")
        st.markdown("### Sequential Steps")
        current_index = workflow_pages.index(st.session_state.page) if st.session_state.page in workflow_pages else 0
        selected_display = st.radio("Select a step:", options=[page_display[p] for p in workflow_pages], index=current_index, key="sidebar_radio")
        selected_page = [p for p, d in page_display.items() if d == selected_display][0]
        if selected_page != st.session_state.page:
            go_to(selected_page)
        if not PYCARET_AVAILABLE:
            st.error("⚠️ PyCaret not installed. Install with: `pip install pycaret` for full AutoML.")
        if st.button("👋🏻 Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.user_email = ""
            reset_ml_state()
            go_to("front")
    # Render page
    if st.session_state.page == "account":
        account_page()
    elif st.session_state.page == "data_upload":
        upload_page()
    elif st.session_state.page == "eda":
        eda_page()
    elif st.session_state.page == "model_training":
        training_page()
    elif st.session_state.page == "model_evaluation":
        evaluation_page()
    elif st.session_state.page == "export_results":
        export_page()
    else:
        upload_page()

# ---------- Main routing ----------
if st.session_state.page == "front":
    front_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "dashboard":
    if not st.session_state.logged_in:
        go_to("login")
    else:
        dashboard_page()
else:
    if not st.session_state.logged_in and st.session_state.page not in ["front", "login"]:
        go_to("login")
    else:
        if st.session_state.logged_in:
            dashboard_page()
        else:
            front_page()