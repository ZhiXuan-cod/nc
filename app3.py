import streamlit as st
import os
import base64
import hashlib
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
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# ---------- Supabase import ----------
from supabase import create_client

# ---------- PyCaret imports (including clustering) ----------
try:
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare, predict_model as clf_predict, get_config, pull
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, predict_model as reg_predict
    from pycaret.clustering import setup as clust_setup, create_model as clust_create, assign_model as clust_assign, pull as clust_pull
    PYCARET_AVAILABLE = True
    CLUSTERING_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    CLUSTERING_AVAILABLE = False
except Exception:
    CLUSTERING_AVAILABLE = False

# ---------- Scipy for outlier detection ----------
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------- Minimal PDF generator ----------
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

# ---------- Background image helper ----------
def get_base64_of_file(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_bg_image_local(image_path):
    bin_str = get_base64_of_file(image_path)
    if bin_str:
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
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

# ---------- Password hashing helpers ----------
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
            candidate_hash = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt,
                iterations,
            )
            return candidate_hash == expected_hash
        except Exception:
            return False

    return stored_password == plain_password

# ---------- Supabase client ----------
if "supabase" not in st.session_state:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        st.session_state.supabase = create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
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

# ---------- Page navigation ----------
PAGES = {
    "front": "Front Page",
    "login": "Login / Register",
    "dashboard": "Dashboard",
    "account": "Account",
    "data_upload": "Data Upload",
    "data_cleaning": "Data Cleaning",
    "eda": "Exploratory Data Analysis",
    "model_training": "Model Training",
    "model_evaluation": "Model Evaluation",
    "export_results": "Export Results"
}

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
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            raise Exception("Streamlit rerun not available. Please upgrade Streamlit.")

st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Global CSS (sidebar colour preserved) ----------
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
    /* Sidebar background colour (preserved) */
    section[data-testid="stSidebar"] {
        background: #ffffe0 !important;
    }
    section[data-testid="stSidebar"] .st-emotion-cache-1wrcr25, 
    section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state initialisation ----------
if "data" not in st.session_state:
    st.session_state.data = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "model" not in st.session_state:
    st.session_state.model = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "test_labels" not in st.session_state:
    st.session_state.test_labels = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "training_done" not in st.session_state:
    st.session_state.training_done = False
# Clustering specific
if "cluster_labels" not in st.session_state:
    st.session_state.cluster_labels = None
if "cluster_metrics" not in st.session_state:
    st.session_state.cluster_metrics = None
if "clustering_model" not in st.session_state:
    st.session_state.clustering_model = None

# ---------- Cleaning helper ----------
def apply_cleaning(df, drop_duplicates, missing_option, outlier_option,
                    encode_option, scale_option, cols_to_drop, target_col):
    cleaned = df.copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    if missing_option != "None":
        if missing_option == "Drop rows with any missing":
            cleaned = cleaned.dropna()
        elif missing_option == "Drop columns with any missing":
            cols_with_na = cleaned.columns[cleaned.isnull().any()].tolist()
            cols_to_drop_na = [c for c in cols_with_na if c != target_col]
            cleaned = cleaned.drop(columns=cols_to_drop_na, errors='ignore')
        elif missing_option == "Fill numeric with mean":
            num_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col != target_col:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
        elif missing_option == "Fill numeric with median":
            num_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col != target_col:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        elif missing_option == "Fill categorical with mode":
            cat_cols = cleaned.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if col != target_col:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0] if not cleaned[col].mode().empty else "Unknown")
    if outlier_option != "None" and SCIPY_AVAILABLE:
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c != target_col]
        if outlier_option == "Remove rows with Z-score > 3":
            if len(num_cols) > 0:
                numeric_subset = cleaned[num_cols].dropna()
                if not numeric_subset.empty:
                    z_scores = np.abs(stats.zscore(numeric_subset, nan_policy='omit'))
                    if np.ndim(z_scores) == 1:
                        z_scores = z_scores.reshape(-1, 1)
                    outlier_rows = (z_scores > 3).any(axis=1)
                    outlier_idx = numeric_subset.index[outlier_rows]
                    cleaned = cleaned.drop(index=outlier_idx)
        elif outlier_option == "Cap at 1st and 99th percentile":
            for col in num_cols:
                q1 = cleaned[col].quantile(0.01)
                q99 = cleaned[col].quantile(0.99)
                cleaned[col] = cleaned[col].clip(lower=q1, upper=q99)
    elif outlier_option != "None" and not SCIPY_AVAILABLE:
        st.warning("Scipy not installed. Z‑score outlier detection disabled.")
    if encode_option != "None":
        cat_cols = cleaned.select_dtypes(include=['object']).columns
        cat_cols = [c for c in cat_cols if c != target_col]
        if len(cat_cols) > 0:
            if encode_option == "Label Encoding":
                for col in cat_cols:
                    le = LabelEncoder()
                    cleaned[col] = le.fit_transform(cleaned[col].astype(str))
            elif encode_option == "One-Hot Encoding":
                cleaned = pd.get_dummies(cleaned, columns=cat_cols, drop_first=True)
    if scale_option != "None":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c != target_col]
        if len(num_cols) > 0:
            if scale_option == "Standardization (z-score)":
                scaler = StandardScaler()
                cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
            elif scale_option == "Normalization (min-max)":
                scaler = MinMaxScaler()
                cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop, errors='ignore')
    return cleaned

# ---------- Safe PyCaret setup ----------
def _pycaret_setup_safe(setup_fn, **kwargs):
    import inspect
    try:
        params = set(inspect.signature(setup_fn).parameters.keys())
    except Exception:
        params = set()
    if params:
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return setup_fn(**filtered)
    try:
        return setup_fn(**kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" in msg:
            import re
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if m:
                bad = m.group(1)
                kwargs.pop(bad, None)
                return _pycaret_setup_safe(setup_fn, **kwargs)
        raise

# ===================== 任务可用性检测函数 =====================
def is_classification_possible(df) -> Tuple[bool, List[str]]:
    """
    检测是否有至少一列适合作为分类目标。
    返回 (是否可能, 候选列列表)
    """
    candidates = []
    for col in df.columns:
        dtype = df[col].dtype
        unique_vals = df[col].nunique(dropna=False)
        if dtype in ['object', 'category']:
            candidates.append(col)
        elif np.issubdtype(dtype, np.number):
            if unique_vals < 20:   # 低基数数值也视为分类候选
                candidates.append(col)
    return len(candidates) > 0, candidates

def is_regression_possible(df) -> Tuple[bool, List[str]]:
    """
    检测是否有至少一列适合作为回归目标（数值列且唯一值较多）。
    返回 (是否可能, 候选列列表)
    """
    candidates = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            # 数值列且不是低基数（避免分类误判）
            if df[col].nunique(dropna=False) >= 20:
                candidates.append(col)
    return len(candidates) > 0, candidates

def is_clustering_possible(df, min_rows=10, min_numeric_features=2) -> Tuple[bool, str]:
    """
    检测数据集是否适合聚类。
    条件：
    - 行数 >= min_rows
    - 数值特征数量 >= min_numeric_features
    - 每个数值特征的方差 > 0（不是常数）
    """
    if len(df) < min_rows:
        return False, f"数据行数不足 {min_rows} 行（当前 {len(df)} 行）"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < min_numeric_features:
        return False, f"数值特征不足 {min_numeric_features} 个（当前 {len(numeric_cols)} 个）"
    # 检查常数特征
    constant_cols = []
    for col in numeric_cols:
        if df[col].var() == 0:
            constant_cols.append(col)
    if constant_cols:
        return False, f"存在常数数值特征: {', '.join(constant_cols[:3])}"
    return True, "适合聚类"

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
    }
    .right-panel h1 { text-shadow: 2px 2px 4px rgba(0,0,0,0.5); font-size: 3rem; margin-bottom: 1rem; }
    .right-panel p { text-shadow: 1px 1px 2px rgba(0,0,0,0.5); font-size: 1.2rem; opacity: 0.9; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        video_path = "animation.mp4"
        if os.path.exists(video_path):
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode()
            video_html = f'<video width="100%" autoplay loop muted playsinline><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video>'
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

# ---------- 增强的 upload_page：动态显示可用任务 ----------
def upload_page():
    st.markdown('<h2 class="sub-header">📁 Upload Your Dataset</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
        <h4>📁 Supported Data Format</h4>
        <ul><li>CSV files only (.csv)</li><li>Structured tabular data</li><li>Numerical and categorical variables</li><li>Clear target column for supervised learning (not required for clustering)</li></ul>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            last_error = None
            for enc in encodings:
                try:
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
            st.session_state.data = df
            st.success(f"✔️ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        if st.session_state.data is not None:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.data.head(), use_container_width=True)
            with st.expander("📊 Basic Data Statistics"):
                df = st.session_state.data
                st.write("**Shape:**", df.shape)
                col_types = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str), 'Missing Values': df.isnull().sum(), 'Unique Values': df.nunique()})
                st.dataframe(col_types, use_container_width=True)

            # ----- 检测可用任务 -----
            classification_possible, class_candidates = is_classification_possible(df)
            regression_possible, reg_candidates = is_regression_possible(df)
            clustering_possible, cluster_msg = is_clustering_possible(df)

            available_tasks = []
            if classification_possible:
                available_tasks.append("Classification")
            if regression_possible:
                available_tasks.append("Regression")
            if clustering_possible:
                available_tasks.append("Clustering")

            if not available_tasks:
                st.error("❌ 当前数据不支持任何机器学习任务（分类、回归、聚类均不适合）。请上传其他数据集。")
                return

            st.markdown("### 🎯 根据数据自动检测到的可用任务")
            st.info(f"**✅ 可以执行的任务：** {', '.join(available_tasks)}")
            if not classification_possible:
                st.warning("⚠️ 分类不可用：没有适合作为分类目标的列（分类目标需要是类别型或低基数数值列）。")
            if not regression_possible:
                st.warning("⚠️ 回归不可用：没有适合作为回归目标的数值列（需要至少一个数值列且唯一值较多）。")
            if not clustering_possible:
                st.warning(f"⚠️ 聚类不可用：{cluster_msg}")

            st.markdown("---")
            st.markdown("### 📌 请选择任务类型")
            problem_type = st.selectbox("Select problem type:", available_tasks)

            if problem_type == "Clustering":
                st.info("Clustering is unsupervised – no target column required.")
                if st.button("Set Clustering Task", type="primary", key="set_clustering"):
                    st.session_state.target_column = None
                    st.session_state.problem_type = "Clustering"
                    st.success("✅ Clustering task selected. No target column needed.")
            else:
                # Classification 或 Regression：显示适合的目标列
                if problem_type == "Classification":
                    candidates = class_candidates
                    help_text = "选择分类目标列（类别型或低基数数值列）"
                else:
                    candidates = reg_candidates
                    help_text = "选择回归目标列（连续数值列）"

                if not candidates:
                    st.error(f"❌ 尽管 {problem_type} 被检测为可能，但未找到具体候选列。请检查数据。")
                    return

                target_col = st.selectbox(f"Select target column for {problem_type}:", candidates, help=help_text)
                if st.button("Set Target", type="primary", key="set_target"):
                    # 额外验证
                    if problem_type == "Classification" and df[target_col].nunique() > 50:
                        st.warning(f"⚠️ 目标列 '{target_col}' 有 {df[target_col].nunique()} 个唯一值，分类效果可能不佳。考虑使用回归。")
                    elif problem_type == "Regression" and not np.issubdtype(df[target_col].dtype, np.number):
                        st.error(f"❌ 目标列 '{target_col}' 不是数值类型，无法进行回归。")
                        return
                    st.session_state.target_column = target_col
                    st.session_state.problem_type = problem_type
                    st.success(f"✅ Target set: {target_col} ({problem_type})")
        else:
            st.info("📂 No data loaded yet. Please upload a CSV file.")
    with col2:
        if st.session_state.data is not None:
            # 显示原有候选列表（仅作参考）
            df = st.session_state.data
            classification_candidates = []
            regression_candidates = []
            for col in df.columns:
                dtype = df[col].dtype
                unique_vals = df[col].nunique(dropna=False)
                if dtype in ['object', 'category']:
                    classification_candidates.append(col)
                elif np.issubdtype(dtype, np.number):
                    if unique_vals < 20:
                        classification_candidates.append(col)
                    else:
                        regression_candidates.append(col)
            st.markdown("### 🔍 检测到的目标候选列（参考）")
            col1a, col2a = st.columns(2)
            with col1a:
                st.markdown("**分类候选**")
                if classification_candidates:
                    st.write(", ".join(classification_candidates))
                else:
                    st.write("无")
            with col2a:
                st.markdown("**回归候选**")
                if regression_candidates:
                    st.write(", ".join(regression_candidates))
                else:
                    st.write("无")
            st.markdown("---")
            st.caption("💡 聚类要求至少2个数值特征且无常数特征。")
        else:
            st.info("请先上传数据。")

# ---------- 其他页面（cleaning, eda, training, evaluation, export, account）与原代码相同 ----------
# 注意：training_page 中的 clustering_training_page 已经修正（无 ignore_low_variance）
# 下面给出完整的 clustering_training_page 和其他函数（保持原样，仅确保参数正确）

def cleaning_page():
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return
    target_col = st.session_state.target_column
    st.markdown('<h2 class="sub-header">🧹 Basic Data Cleaning</h2>', unsafe_allow_html=True)
    original_df = st.session_state.data
    st.markdown("### Original Data Preview")
    st.dataframe(original_df.head())
    st.markdown(f"Shape: {original_df.shape}")
    with st.expander("Basic Cleaning Options", expanded=True):
        drop_duplicates = st.checkbox("Drop duplicate rows")
        missing_option = st.selectbox("Handle missing values", ["None", "Drop rows with any missing", "Drop columns with any missing", "Fill numeric with mean", "Fill numeric with median", "Fill categorical with mode"])
        outlier_option = st.selectbox("Handle outliers (numerical columns)", ["None", "Remove rows with Z-score > 3", "Cap at 1st and 99th percentile"])
        cols_to_drop = st.multiselect("Select columns to drop", [c for c in original_df.columns if c != target_col] if target_col else original_df.columns.tolist())
        if st.button("🔍 Preview Cleaning", type="secondary", key="preview_cleaning"):
            cleaned = apply_cleaning(original_df, drop_duplicates, missing_option, outlier_option, encode_option="None", scale_option="None", cols_to_drop=cols_to_drop, target_col=target_col)
            st.markdown("### Cleaned Data Preview")
            st.dataframe(cleaned.head())
            st.markdown(f"Final shape: {cleaned.shape}")
            st.session_state.cleaned_data = cleaned
    if st.session_state.cleaned_data is not None:
        if st.button("✅ Apply Cleaning", type="primary", key="apply_cleaning"):
            cleaned = apply_cleaning(original_df, drop_duplicates, missing_option, outlier_option, encode_option="None", scale_option="None", cols_to_drop=cols_to_drop, target_col=target_col)
            st.session_state.data = cleaned
            st.session_state.cleaned_data = None
            st.success("Data cleaned successfully!")

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
        st.dataframe(missing_df, use_container_width=True)
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
            st.dataframe(col_stats, use_container_width=True)
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

# ---------- Clustering training function (FIXED: removed ignore_low_variance) ----------
def clustering_training_page():
    """Train a clustering model using PyCaret's clustering module."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return
    if not PYCARET_AVAILABLE or not CLUSTERING_AVAILABLE:
        st.error("PyCaret clustering module not available. Please install PyCaret (pip install pycaret) and ensure it includes clustering.")
        return

    st.markdown('<h2 class="sub-header">🎯 Automated Clustering (Unsupervised)</h2>', unsafe_allow_html=True)
    df = st.session_state.data.copy()

    st.markdown(f"""
    <div class="card">
    <h4>Clustering Configuration</h4>
    <ul><li><strong>Dataset Shape:</strong> {df.shape}</li><li><strong>Unsupervised – no target column</strong></li></ul>
    </div>
    """, unsafe_allow_html=True)

    # Clustering algorithm selection
    model_list = ['kmeans', 'hclust', 'dbscan', 'birch', 'meanshift', 'optics']
    selected_model = st.selectbox("Select clustering algorithm", model_list, index=0)
    num_clusters = st.slider("Number of clusters (for k-means/hclust)", 2, 15, 3)
    normalize = st.checkbox("Normalize data (recommended for distance-based algorithms)", value=True)

    # Additional parameters for DBSCAN
    eps = None
    min_samples = None
    if selected_model == 'dbscan':
        eps = st.slider("Epsilon (eps) – neighbourhood radius", 0.1, 2.0, 0.5, 0.05)
        min_samples = st.slider("Minimum samples per cluster", 2, 20, 5)

    if st.button("🚀 Train Clustering Model", type="primary"):
        with st.spinner(f"Training {selected_model} clustering model..."):
            try:
                # ✅ 仅使用 clust_setup 支持的参数
                setup_args = {
                    "data": df,
                    "normalize": normalize,
                    "session_id": 42,
                    "verbose": False,
                    "log_experiment": False
                }
                clust_setup(**setup_args)
                st.toast("PyCaret setup complete", icon="✅")

                # Create model
                if selected_model == 'dbscan':
                    model = clust_create(selected_model, eps=eps, min_samples=min_samples)
                else:
                    model = clust_create(selected_model, num_clusters=num_clusters)

                # Assign clusters to data
                assigned = clust_assign(model)
                cluster_labels = assigned['Cluster'].values

                # Get metrics
                metrics_df = clust_pull()
                silhouette = None
                if metrics_df is not None and not metrics_df.empty:
                    if 'Silhouette' in metrics_df.columns:
                        silhouette = metrics_df['Silhouette'].values[0]
                    elif 'Silhouette Score' in metrics_df.columns:
                        silhouette = metrics_df['Silhouette Score'].values[0]

                # Store results
                st.session_state.cluster_labels = cluster_labels
                st.session_state.clustering_model = model
                st.session_state.training_complete = True
                st.session_state.training_done = True
                st.session_state.problem_type = "Clustering"

                st.session_state.cluster_metrics = {
                    "algorithm": selected_model,
                    "num_clusters": len(np.unique(cluster_labels)),
                    "silhouette_score": silhouette,
                    "cluster_sizes": pd.Series(cluster_labels).value_counts().to_dict()
                }

                st.success("🎉 Clustering completed successfully!")
                with st.expander("📊 Clustering Results", expanded=True):
                    st.markdown("#### Model Information")
                    st.code(str(model), language='python')
                    if silhouette is not None:
                        st.metric("Silhouette Score", f"{silhouette:.4f}")
                    else:
                        st.info("Silhouette score not available in metrics output.")
                    st.markdown("#### Cluster Sizes")
                    sizes_df = pd.DataFrame(list(st.session_state.cluster_metrics["cluster_sizes"].items()), columns=["Cluster", "Count"])
                    fig = px.bar(sizes_df, x="Cluster", y="Count", title="Number of points per cluster")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Clustering failed: {type(e).__name__}: {str(e)}")
                st.exception(e)

# ---------- Main training page (dispatcher) ----------
def training_page():
    # Route to clustering if problem type is Clustering
    if st.session_state.problem_type == "Clustering":
        clustering_training_page()
        return

    # Otherwise original classification/regression training
    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload data and set target column first.")
        return
    st.markdown('<h2 class="sub-header">📐 Automated Model Training (PyCaret)</h2>', unsafe_allow_html=True)
    if not PYCARET_AVAILABLE:
        st.error("⚠️ PyCaret not installed. Install with `pip install pycaret` to use AutoML.")
        return
    df = st.session_state.data.copy()
    target_col = st.session_state.target_column
    problem_type = st.session_state.problem_type
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' is not in the dataset.")
        return
    if df[target_col].isnull().sum() > 0:
        st.error(f"Target column '{target_col}' contains missing values. Please handle them in Data Cleaning.")
        return
    if problem_type == "Regression":
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            st.error(f"❌ Target column '{target_col}' must be numeric for regression. Current type: {df[target_col].dtype}. Please select a numeric column or change the problem type to Classification.")
            return
        if np.isinf(df[target_col]).any():
            st.error("❌ Target column contains infinite values. Please remove or replace them.")
            return
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            st.warning(f"Feature column '{col}' contains infinite values. They may cause training errors. Consider cleaning them.")
    if problem_type == "Classification" and df[target_col].nunique() > 20:
        st.warning(f"Target column has {df[target_col].nunique()} unique values. Classification may be slow or have low accuracy. Consider regression or reduce categories.")
    if len(df) < 20:
        st.warning("⚠️ Dataset has very few rows (<20). Model may not generalize well.")
    st.markdown(f"""
    <div class="card">
    <h4>Training Configuration</h4>
    <ul><li><strong>Problem Type:</strong> {problem_type}</li><li><strong>Target Column:</strong> {target_col}</li><li><strong>Dataset Shape:</strong> {df.shape}</li></ul>
    </div>
    """, unsafe_allow_html=True)
    if "training_mode" not in st.session_state:
        st.session_state.training_mode = "Balanced"
    col_mode, _ = st.columns([1, 2])
    with col_mode:
        mode = st.selectbox("Training Mode", ["Fast", "Balanced", "Accurate"], index=["Fast", "Balanced", "Accurate"].index(st.session_state.training_mode), help="Fast: lightweight models only; Balanced: mix of models; Accurate: more models & deeper tuning")
    if mode != st.session_state.training_mode:
        st.session_state.training_mode = mode
    if problem_type == "Classification":
        if mode == "Fast":
            allowed_models = ['lr', 'ridge', 'dt']
        elif mode == "Balanced":
            allowed_models = ['lr', 'ridge', 'dt', 'rf', 'nb']
        else:
            allowed_models = ['lr', 'ridge', 'dt', 'rf', 'nb', 'svm', 'xgboost']
    else:
        if mode == "Fast":
            allowed_models = ['lr', 'ridge', 'dt']
        elif mode == "Balanced":
            allowed_models = ['lr', 'ridge', 'dt', 'rf', 'lar']
        else:
            allowed_models = ['lr', 'ridge', 'dt', 'rf', 'lar', 'svm', 'xgboost']
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    with col2:
        fold = st.slider("Cross-validation folds", 3, 10, 5)
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)
    sample_frac = st.slider("Sample fraction (optional, for speed)", 0.1, 1.0, 1.0, 0.05)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        st.info(f"Using {len(df)} rows after sampling (original: {st.session_state.data.shape[0]} rows).")
    if st.button("🚀 Start Automated Training", type="primary", key="start_training"):
        with st.spinner(f"🧠 PyCaret is training {len(allowed_models)} models with {fold}-fold CV..."):
            st.toast("Training started...")
            try:
                setup_args = {"data": df, "target": target_col, "train_size": 1 - test_size, "session_id": random_state, "fold": fold, "n_jobs": 1, "html": False, "verbose": False, "preprocess": True, "ignore_low_variance": False, "remove_multicollinearity": False, "log_experiment": False}
                if problem_type == "Classification":
                    setup_args["data_split_stratify"] = True
                    _pycaret_setup_safe(clf_setup, **setup_args)
                else:
                    setup_args["data_split_stratify"] = False
                    _pycaret_setup_safe(reg_setup, **setup_args)
                try:
                    X_train = get_config('X_train')
                    st.session_state.feature_names = X_train.columns.tolist()
                except Exception as e:
                    st.error("❌ PyCaret setup failed. Check data format or target column.")
                    st.exception(e)
                    return
                if problem_type == "Classification":
                    best_model = clf_compare(include=allowed_models, n_select=1, verbose=False, sort='Accuracy')
                else:
                    best_model = reg_compare(include=allowed_models, n_select=1, verbose=False, sort='R2')
                if problem_type == "Classification":
                    pred_df = clf_predict(best_model)
                else:
                    pred_df = reg_predict(best_model)
                if problem_type == "Classification":
                    test_predictions = pred_df.get("prediction_label", pred_df.iloc[:, -1])
                else:
                    test_predictions = pred_df.get("prediction_label", pred_df.iloc[:, -1])
                y_test = pred_df[target_col]
                st.session_state.predictions = test_predictions.values
                st.session_state.test_labels = y_test.values
                st.session_state.model = best_model
                st.session_state.training_complete = True
                st.session_state.training_done = True
                with st.expander("📊 Training Results (click to expand)", expanded=True):
                    st.markdown("#### 🏆 Best Model")
                    st.code(str(best_model), language='python')
                    if hasattr(best_model, 'feature_importances_'):
                        st.markdown("#### 🔍 Feature Importance (Top 10)")
                        feature_names = st.session_state.feature_names
                        importances = best_model.feature_importances_
                        if feature_names and len(feature_names) == len(importances):
                            imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                            imp_df = imp_df.sort_values('importance', ascending=False).head(10)
                            st.dataframe(imp_df, use_container_width=True)
                        else:
                            st.warning(f"Feature importance length ({len(importances)}) does not match feature names length ({len(feature_names)}). Skipping display.")
                    elif hasattr(best_model, 'coef_'):
                        st.markdown("#### 🔍 Model Coefficients")
                        feature_names = st.session_state.feature_names
                        coef = best_model.coef_
                        if coef.ndim == 2:
                            imp_df = pd.DataFrame({'feature': feature_names, 'importance': np.abs(coef).mean(axis=0)})
                            imp_df = imp_df.sort_values('importance', ascending=False).head(10)
                            st.markdown("Mean Absolute Coefficients (Multi-class)")
                        else:
                            imp_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
                            imp_df = imp_df.sort_values('coefficient', ascending=False).head(10)
                            st.markdown("Coefficients (Linear Model)")
                        st.dataframe(imp_df, use_container_width=True)
                    else:
                        st.info("This model does not support feature importance or coefficient display.")
                    comparison_df = pull()
                    if comparison_df is not None:
                        st.markdown("#### 📊 Model Comparison (Top 10)")
                        st.dataframe(comparison_df.head(10), use_container_width=True)
                st.success("🎉 Model training completed successfully!")
            except Exception as e:
                st.error(f"❌ Training failed: {type(e).__name__}: {str(e)}")
                st.exception(e)

def evaluation_page():
    if not st.session_state.training_complete:
        st.warning("⚠️ No trained model found. Please go to 'Model Training' and train a model first.")
        return

    # Clustering evaluation
    if st.session_state.problem_type == "Clustering":
        st.markdown('<h2 class="sub-header">📈 Clustering Performance Evaluation</h2>', unsafe_allow_html=True)
        if st.session_state.cluster_labels is None:
            st.error("❌ No clustering labels found. Please retrain the clustering model.")
            return

        labels = st.session_state.cluster_labels
        df = st.session_state.data.copy()
        df['Cluster'] = labels

        # Use only numeric columns for metrics
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            try:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                sil_score = silhouette_score(numeric_df, labels)
                ch_score = calinski_harabasz_score(numeric_df, labels)
                db_score = davies_bouldin_score(numeric_df, labels)
                col1, col2, col3 = st.columns(3)
                col1.metric("Silhouette Score", f"{sil_score:.4f}")
                col2.metric("Calinski-Harabasz Index", f"{ch_score:.2f}")
                col3.metric("Davies-Bouldin Index", f"{db_score:.4f}")
            except Exception as e:
                st.warning(f"Could not compute clustering metrics: {e}")
        else:
            st.info("Not enough numeric columns to compute clustering metrics.")

        # Cluster distribution
        st.markdown("### Cluster Distribution")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values, labels={'x': 'Cluster', 'y': 'Count'}, title="Number of points per cluster")
        st.plotly_chart(fig, use_container_width=True)

        # 2D PCA visualisation
        if numeric_df.shape[1] >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(numeric_df)
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = labels.astype(str)
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="PCA Projection of Clusters")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least two numeric features for PCA visualisation.")

        # Show cluster assignments
        st.markdown("### Cluster Assignments (first 100 rows)")
        st.dataframe(df[['Cluster'] + [c for c in df.columns if c != 'Cluster']].head(100), use_container_width=True)
        return

    # ---------- Original evaluation for classification/regression ----------
    st.markdown('<h2 class="sub-header">📈 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    if st.session_state.predictions is None or st.session_state.test_labels is None:
        st.error("❌ Model predictions or test labels missing. Please retrain the model.")
        return
    model = st.session_state.model
    predictions = st.session_state.predictions
    y_test = st.session_state.test_labels
    problem_type = st.session_state.problem_type
    y_test = np.asarray(y_test).ravel()
    predictions = np.asarray(predictions).ravel()
    if pd.isnull(y_test).any() or pd.isnull(predictions).any():
        st.error("Test data or predictions contain NaN values. Cannot compute metrics.")
        return
    with st.expander("🔍 Model Information", expanded=False):
        st.markdown("#### Best Model")
        st.code(str(model), language='python')
        try:
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### 📊 Feature Importance (All Features)")
                feature_names = st.session_state.get('feature_names', None)
                importances = model.feature_importances_
                if feature_names and len(feature_names) == len(importances):
                    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
                    st.dataframe(imp_df, use_container_width=True)
                    fig = px.bar(imp_df.head(15), x='Importance', y='Feature', orientation='h', title="Top 15 Feature Importance")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Feature names not available or length mismatch.")
            elif hasattr(model, 'coef_'):
                feature_names = st.session_state.get('feature_names', None)
                coef = model.coef_
                if feature_names:
                    if coef.ndim == 2:
                        imp_df = pd.DataFrame({'feature': feature_names, 'importance': np.abs(coef).mean(axis=0)}).sort_values('importance', ascending=False)
                        st.markdown("#### Mean Absolute Coefficients (Multi-class)")
                    else:
                        imp_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef}).sort_values('coefficient', ascending=False)
                    st.dataframe(imp_df, use_container_width=True)
                else:
                    st.info("Feature names not available.")
            else:
                st.info("This model does not support feature importance display.")
        except Exception as e:
            st.warning(f"Feature importance display failed: {e}")
    if problem_type == "Classification":
        st.markdown("### Classification Metrics")
        try:
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
            rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        except Exception as e:
            st.error(f"Error computing classification metrics: {e}")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{acc:.4f}")
            st.metric("Precision (weighted)", f"{prec:.4f}")
        with col2:
            st.metric("Recall (weighted)", f"{rec:.4f}")
            st.metric("F1 Score (weighted)", f"{f1:.4f}")
        try:
            cm = confusion_matrix(y_test, predictions)
            fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate confusion matrix: {e}")
        try:
            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate classification report: {e}")
    else:
        st.markdown("### Regression Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2_score(y_test, predictions):.4f}")
            st.metric("MAE", f"{mean_absolute_error(y_test, predictions):.4f}")
        with col2:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, predictions)):.4f}")
            mask = y_test != 0
            if mask.any():
                mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
                st.metric("MAPE (%)", f"{mape:.2f}")
            else:
                st.metric("MAPE (%)", "N/A (zero values present)")
        residuals = y_test - predictions
        fig = px.scatter(x=predictions, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title="Residuals vs Predicted")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal', line=dict(dash='dash', color='red')))
        st.plotly_chart(fig, use_container_width=True)

def export_page():
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first to export results.")
        return
    st.markdown('<h2 class="sub-header">💾 Export Model and Results</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Model Information")
        if st.button("Show Model Details", key="show_model_details"):
            if st.session_state.problem_type == "Clustering":
                st.write("**Clustering Model:**", st.session_state.clustering_model)
            else:
                st.write("**Best Model:**", st.session_state.model)
    with col2:
        st.markdown("#### 📥 Download Predictions / Cluster Assignments")
        if st.session_state.problem_type == "Clustering" and st.session_state.cluster_labels is not None:
            results_df = st.session_state.data.copy()
            results_df["Cluster"] = st.session_state.cluster_labels
            st.download_button(label="Download Cluster Assignments (CSV)", data=results_df.to_csv(index=False), file_name="cluster_assignments.csv", mime="text/csv", key="export_clusters")
        elif st.session_state.predictions is not None and st.session_state.test_labels is not None:
            results_df = pd.DataFrame({"Actual": st.session_state.test_labels, "Predicted": st.session_state.predictions})
            st.download_button(label="Download Predictions (CSV)", data=results_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv", key="export_predictions")
        else:
            st.warning("No predictions or cluster labels available. Please retrain the model.")
    st.markdown("#### 📄 Model Report")
    if st.button("Generate Model Report", key="generate_report"):
        if st.session_state.data is not None:
            dataset_shape = st.session_state.data.shape
            feature_count = len(st.session_state.data.columns) - (1 if st.session_state.target_column else 0)
        else:
            dataset_shape = "N/A"
            feature_count = "N/A"
        if st.session_state.problem_type == "Clustering":
            report_content = f"""
# Clustering Model Report

## Project Information
- Platform: No-Code ML Platform
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Problem Type: Clustering (Unsupervised)
- Algorithm: {st.session_state.cluster_metrics.get('algorithm', 'N/A') if st.session_state.cluster_metrics else 'N/A'}

## Dataset Information
- Original Shape: {dataset_shape}
- Features used: {feature_count}

## Clustering Results
- Number of clusters: {st.session_state.cluster_metrics.get('num_clusters', 'N/A') if st.session_state.cluster_metrics else 'N/A'}
- Silhouette Score: {st.session_state.cluster_metrics.get('silhouette_score', 'N/A') if st.session_state.cluster_metrics else 'N/A'}

## Notes
This clustering was generated using PyCaret AutoML through the No-Code ML Platform.
"""
        else:
            report_content = f"""
# Machine Learning Model Report

## Project Information
- Platform: No-Code ML Platform
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Problem Type: {st.session_state.problem_type}
- Target Column: {st.session_state.target_column}

## Dataset Information
- Original Shape: {dataset_shape}
- Features: {feature_count}

## Model Information
- Best Model: {st.session_state.model}
- Training Completed: {st.session_state.training_complete}

## Notes
This model was generated using PyCaret AutoML through the No-Code ML Platform.
"""
        st.code(report_content, language='markdown')
        pdf_bytes = text_to_simple_pdf_bytes(report_content, title="ML Model Report")
        st.download_button("📥 Download Report (PDF)", data=pdf_bytes, file_name="ml_model_report.pdf", mime="application/pdf", key="download_pdf")
        st.download_button("📥 Download Report (Markdown)", data=report_content, file_name="ml_model_report.md", mime="text/markdown", key="download_md")
    st.markdown("### 📋 Session Information")
    session_info = {
        "Data Loaded": st.session_state.data is not None,
        "Problem Type": st.session_state.problem_type,
        "Target Column": st.session_state.target_column if st.session_state.target_column else "N/A (Clustering)",
        "Model Trained": st.session_state.training_complete,
        "Predictions/Clusters Available": (st.session_state.predictions is not None) or (st.session_state.cluster_labels is not None)
    }
    session_df = pd.DataFrame.from_dict(session_info, orient='index', columns=['Status'])
    st.dataframe(session_df, use_container_width=True)
    if st.button("🔄 Start Over", type="secondary", key="start_over"):
        keys_to_reset = ["data", "target_column", "problem_type", "model", "predictions", "test_labels", "training_complete", "cleaned_data", "feature_names", "training_done", "cluster_labels", "cluster_metrics", "clustering_model"]
        for key in keys_to_reset:
            if key in st.session_state:
                st.session_state[key] = None
        go_to("data_upload")

def account_page():
    st.markdown('<h2 class="sub-header">👤 Account Settings</h2>', unsafe_allow_html=True)
    st.markdown("### Your Profile")
    st.write(f"**Name:** {st.session_state.user_name}")
    st.write(f"**Email:** {st.session_state.user_email}")
    st.markdown("---")
    st.markdown("### Change Password")
    with st.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_new = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Update Password")
        if submitted:
            if not current_password or not new_password or not confirm_new:
                st.error("Please fill in all fields.")
            elif new_password != confirm_new:
                st.error("New passwords do not match.")
            elif len(new_password) < 6:
                st.error("New password must be at least 6 characters.")
            else:
                if st.session_state.supabase is None:
                    st.error("Supabase connection is not available. Cannot update password.")
                else:
                    try:
                        response = st.session_state.supabase.table("users").select("*").eq("email", st.session_state.user_email).execute()
                        if len(response.data) == 0:
                            st.error("User not found.")
                        else:
                            user = response.data[0]
                            stored_hash = user.get("password", "")
                            if verify_password(current_password, stored_hash):
                                new_hash = hash_password(new_password)
                                st.session_state.supabase.table("users").update({"password": new_hash}).eq("email", st.session_state.user_email).execute()
                                st.success("Password updated successfully!")
                            else:
                                st.error("Current password is incorrect.")
                    except Exception as e:
                        st.error(f"Failed to update password: {e}")

def dashboard_page():
    set_bg_image_local("purple.png")
    st.markdown(f"<h1 style='color: black;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)
    workflow_pages = ["data_upload", "data_cleaning", "eda", "model_training", "model_evaluation", "export_results", "account"]
    page_display = {
        "data_upload": "📁 Data Upload",
        "data_cleaning": "🧹 Data Cleaning",
        "eda": "🔍 Exploratory Data Analysis",
        "model_training": "📐 Model Training",
        "model_evaluation": "📈 Model Evaluation",
        "export_results": "💾 Export Results",
        "account": "👤 Account Settings"
    }
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
        st.markdown("---")
        st.markdown("### Sequential Steps")
        if st.session_state.page in workflow_pages:
            current_index = workflow_pages.index(st.session_state.page)
        else:
            current_index = 0
        selected_display = st.radio("Select a step:", options=[page_display[p] for p in workflow_pages], index=current_index, key="sidebar_radio")
        selected_page = [p for p, d in page_display.items() if d == selected_display][0]
        if selected_page != st.session_state.page:
            go_to(selected_page)
        if not PYCARET_AVAILABLE:
            st.error("⚠️ PyCaret not installed. Install with: `pip install pycaret`")
            st.code("pip install pycaret", language="bash")
        if st.button("👋🏻 Logout", type="primary", key="logout_button"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.user_email = ""
            keys_to_clear = ["data", "target_column", "problem_type", "model", "predictions", "test_labels", "training_complete", "cleaned_data", "feature_names", "training_done", "cluster_labels", "cluster_metrics", "clustering_model"]
            for key in keys_to_clear:
                if key in st.session_state:
                    st.session_state[key] = None
            go_to("front")
    # Render the appropriate page
    if st.session_state.page == "account":
        account_page()
    elif st.session_state.page == "data_upload":
        upload_page()
    elif st.session_state.page == "data_cleaning":
        cleaning_page()
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