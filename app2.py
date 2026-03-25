import streamlit as st
import os
import base64
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

from supabase import create_client

# ---------- Minimal PDF generator (no extra installs) ----------
def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def text_to_simple_pdf_bytes(text: str, title: str = "ML Model Report") -> bytes:
    """
    Create a simple multi-page PDF containing the given plain text.
    No third-party dependencies (works on Streamlit free tier).
    """
    page_w, page_h = 612, 792  # US Letter points
    margin_x, margin_y = 54, 54
    font_size = 10
    leading = 14
    max_lines = int((page_h - 2 * margin_y) / leading)

    lines = (text or "").splitlines() or ["(empty report)"]
    pages = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]

    objects: List[bytes] = []

    def add_obj(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)  # 1-based object number

    # 1) Catalog (points to Pages obj #2)
    catalog_obj_num = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")

    # Reserve slot for Pages object (#2); fill later
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

# ---------- FLAML import with fallback ----------
try:
    from flaml import AutoML
    flaml_available = True
except ImportError:
    flaml_available = False
    st.warning("⚠️ FLAML not installed. Install with 'pip install flaml[automl]' to use auto‑ML.")
    # Dummy class to avoid NameError
    class AutoML:
        def __init__(self, **kwargs):
            raise ImportError("FLAML is not installed.")
        def fit(self, X, y, **kwargs):
            pass
        def predict(self, X):
            return None
        def score(self, X, y):
            return 0.0

# ---------- Page configuration ----------
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Supabase client ----------
if "supabase" not in st.session_state:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        st.session_state.supabase = create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        st.session_state.supabase = None

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

    # Legacy plain-text fallback
    return stored_password == plain_password

# ---------- User data storage (Supabase) ----------
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
        return False, None
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) == 0:
            return False, None
        user = response.data[0]
        if verify_password(password, user.get("password", "")):
            return True, user["name"]
        return False, None
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False, None

# ---------- Page navigation ----------
if "page" not in st.session_state:
    st.session_state.page = "front"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

def go_to(page):
    st.session_state.page = page

# ---------- Global CSS styles ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3949AB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: rgba(255,255,255,0.85);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    div.stButton > button {
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
    div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Front page ----------
def front_page():
    set_bg_image_local("FrontPage.jpg")
    st.markdown("""
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
    .right-panel h1 {
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .right-panel p {
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-size: 1.2rem;
        opacity: 0.9;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
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
            video_html = f"""
            <video width="100%" autoplay loop muted playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 2rem; text-align: center;">
                <span style="font-size: 3rem;">📹</span>
                <p style="color: white;">Video not found. Please add animation.mp4</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        right_html = """
        <div class="right-panel">
            <h1>Welcome to<br>No-Code ML Platform</h1>
            <p>Accessible Machine Learning without code.</p>
        </div>
        """
        st.markdown(right_html, unsafe_allow_html=True)
        if st.button("Get Started", key="get_started", use_container_width=True):
            go_to("login")
            st.rerun()

# ---------- Login/Register page ----------
def login_page():
    set_bg_image_local("login.jpg")
    st.markdown("""
    <style>
    .stApp {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: white;
        border-bottom-color: #2196F3;
    }
    .stTextInput input {
        color: black !important;
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 5px;
    }
    .stTextInput label {
        color: black !important;
    }
    .back-button-container {
        text-align: center;
        margin-top: 1.5rem;
    }
    .back-button-container button {
        background: transparent !important;
        color: rgba(255,255,255,0.9) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 50px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        width: auto !important;
        display: inline-block !important;
        box-shadow: none !important;
    }
    .back-button-container button:hover {
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.6) !important;
        transform: scale(1.02) !important;
    }
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
                    success, name = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_name = name
                        go_to("dashboard")
                        st.rerun()
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
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Initialize session state ----------
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
if "test_data" not in st.session_state:
    st.session_state.test_data = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "app_page" not in st.session_state:
    st.session_state.app_page = "📁 Data Upload"
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None

# ---------- Dashboard subpages ----------
def upload_page():
    st.markdown('<h2 class="sub-header">📁 Upload Your Dataset</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
        <h4>📁 Supported Data Format</h4>
        <ul>
            <li>CSV files only (.csv)</li>
            <li>Structured tabular data</li>
            <li>Numerical and categorical variables</li>
            <li>Clear target column for supervised learning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"✔️ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                st.markdown("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                with st.expander("📊 Basic Data Statistics"):
                    st.write("**Shape:**", df.shape)
                    col_types = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Missing Values': df.isnull().sum(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(col_types, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Important Notes</h4>
        <ul>
            <li>Ensure your data is clean</li>
            <li>Remove sensitive information</li>
            <li>Check for missing values</li>
            <li>Define target variable clearly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.data is not None:
            st.markdown("### 📌 Define Target Column")
            target_col = st.selectbox(
                "Select the target column:",
                options=st.session_state.data.columns.tolist(),
                index=len(st.session_state.data.columns)-1
            )
            problem_type = st.selectbox("Select problem type:", ["Classification", "Regression"])
            if st.button("Set Target & Continue", type="primary"):
                st.session_state.target_column = target_col
                st.session_state.problem_type = problem_type
                st.success(f"✅ Target set: {target_col} ({problem_type})")
                st.session_state.app_page = "🧹 Data Cleaning"
                st.rerun()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.data is not None and st.session_state.target_column is not None:
            if st.button("➡️ Go to Data Cleaning", type="primary", use_container_width=True):
                st.session_state.app_page = "🧹 Data Cleaning"
                st.rerun()
        else:
            st.button("➡️ Go to Data Cleaning (set target first)", disabled=True, use_container_width=True)

def cleaning_page():
    st.markdown('<h2 class="sub-header">🧹 Data Cleaning</h2>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first from the 'Data Upload' page.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()
        return

    original_df = st.session_state.data
    st.markdown("### Original Data Preview")
    st.dataframe(original_df.head())
    st.markdown(f"Shape: {original_df.shape}")

    with st.expander("Cleaning Options", expanded=True):
        drop_duplicates = st.checkbox("Drop duplicate rows")
        missing_option = st.selectbox(
            "Handle missing values",
            ["None", "Drop rows with any missing", "Drop columns with any missing",
             "Fill numeric with mean", "Fill numeric with median", "Fill categorical with mode"]
        )
        outlier_option = st.selectbox(
            "Handle outliers (numerical columns)",
            ["None", "Remove rows with Z-score > 3", "Cap at 1st and 99th percentile"]
        )
        encode_option = st.selectbox(
            "Categorical encoding",
            ["None", "Label Encoding", "One-Hot Encoding"]
        )
        scale_option = st.selectbox(
            "Feature scaling (numerical)",
            ["None", "Standardization (z-score)", "Normalization (min-max)"]
        )
        cols_to_drop = st.multiselect("Select columns to drop", original_df.columns.tolist())

        if st.button("🔍 Preview Cleaning", type="secondary"):
            cleaned = original_df.copy()

            if drop_duplicates:
                cleaned = cleaned.drop_duplicates()
                st.info(f"Dropped duplicates, new shape: {cleaned.shape}")

            if missing_option != "None":
                if missing_option == "Drop rows with any missing":
                    cleaned = cleaned.dropna()
                    st.info(f"Dropped rows with missing values, new shape: {cleaned.shape}")
                elif missing_option == "Drop columns with any missing":
                    cleaned = cleaned.dropna(axis=1)
                    st.info(f"Dropped columns with missing values, new shape: {cleaned.shape}")
                elif missing_option == "Fill numeric with mean":
                    num_cols = cleaned.select_dtypes(include=[np.number]).columns
                    for col in num_cols:
                        cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                    st.info("Filled numeric missing values with mean.")
                elif missing_option == "Fill numeric with median":
                    num_cols = cleaned.select_dtypes(include=[np.number]).columns
                    for col in num_cols:
                        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                    st.info("Filled numeric missing values with median.")
                elif missing_option == "Fill categorical with mode":
                    cat_cols = cleaned.select_dtypes(include=['object']).columns
                    for col in cat_cols:
                        cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0] if not cleaned[col].mode().empty else "Unknown")
                    st.info("Filled categorical missing values with mode.")

            if outlier_option != "None":
                num_cols = cleaned.select_dtypes(include=[np.number]).columns
                if outlier_option == "Remove rows with Z-score > 3":
                    from scipy import stats
                    if len(num_cols) == 0:
                        st.warning("No numerical columns found for outlier removal.")
                    else:
                        numeric_subset = cleaned[num_cols].dropna()
                        if numeric_subset.empty:
                            st.warning("No complete numerical rows available.")
                        else:
                            z_scores = np.abs(stats.zscore(numeric_subset, nan_policy='omit'))
                            if np.ndim(z_scores) == 1:
                                z_scores = z_scores.reshape(-1, 1)
                            outlier_rows = (z_scores > 3).any(axis=1)
                            outlier_idx = numeric_subset.index[outlier_rows]
                            cleaned = cleaned.drop(index=outlier_idx)
                            st.info(f"Removed rows with Z-score > 3, new shape: {cleaned.shape}")
                elif outlier_option == "Cap at 1st and 99th percentile":
                    for col in num_cols:
                        q1 = cleaned[col].quantile(0.01)
                        q99 = cleaned[col].quantile(0.99)
                        cleaned[col] = cleaned[col].clip(lower=q1, upper=q99)
                    st.info("Capped outliers at 1st and 99th percentiles.")

            if encode_option != "None":
                cat_cols = cleaned.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    if encode_option == "Label Encoding":
                        for col in cat_cols:
                            le = LabelEncoder()
                            cleaned[col] = le.fit_transform(cleaned[col].astype(str))
                        st.info("Applied Label Encoding.")
                    elif encode_option == "One-Hot Encoding":
                        cleaned = pd.get_dummies(cleaned, columns=cat_cols, drop_first=True)
                        st.info("Applied One-Hot Encoding.")

            if scale_option != "None":
                num_cols = cleaned.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    if scale_option == "Standardization (z-score)":
                        scaler = StandardScaler()
                        cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
                        st.info("Applied Standardization.")
                    elif scale_option == "Normalization (min-max)":
                        scaler = MinMaxScaler()
                        cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
                        st.info("Applied Min-Max Normalization.")

            if cols_to_drop:
                cleaned = cleaned.drop(columns=cols_to_drop)
                st.info(f"Dropped selected columns, new shape: {cleaned.shape}")

            st.markdown("### Cleaned Data Preview")
            st.dataframe(cleaned.head())
            st.markdown(f"Final shape: {cleaned.shape}")
            st.session_state.cleaned_data = cleaned

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.cleaned_data is not None:
            if st.button("✅ Apply Cleaning and Continue", type="primary", use_container_width=True):
                st.session_state.data = st.session_state.cleaned_data
                st.session_state.cleaned_data = None
                st.success("Data cleaned successfully!")
                st.session_state.app_page = "🔍 Exploratory Data Analysis"
                st.rerun()
        else:
            st.button("✅ Apply Cleaning (preview first)", disabled=True, use_container_width=True)

def eda_page():
    st.markdown('<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first from the 'Data Upload' page.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()
        return
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
    dtype_df = pd.DataFrame({
        'Data Type': dtype_counts.index.astype(str),
        'Count': dtype_counts.values
    })
    fig = px.pie(dtype_df, values='Count', names='Data Type', title="Distribution of Data Types")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚠️ Missing Values Analysis")
    missing_series = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_series.index,
        'Missing_Count': missing_series.values,
        'Missing_Percentage': (missing_series.values / len(df)) * 100
    }).sort_values('Missing_Percentage', ascending=False)
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    if len(missing_df) > 0:
        fig = px.bar(missing_df, x='Column', y='Missing_Percentage',
                    title="Missing Values by Column (%)", color='Missing_Percentage')
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
            stats = df[selected_num_col].describe()
            st.dataframe(stats, use_container_width=True)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        st.markdown("### 📊 Categorical Columns Analysis")
        selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
        if selected_cat_col:
            value_counts = df[selected_cat_col].value_counts().head(20)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                            title=f"Top Categories in {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(names=value_counts.index, values=value_counts.values,
                            title=f"Distribution of {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)

    if len(numerical_cols) > 1:
        st.markdown("### 🔗 Correlation Matrix")
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        title="Correlation Heatmap")
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

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.target_column is not None:
            if st.button("➡️ Go to Model Training", type="primary", use_container_width=True):
                st.session_state.app_page = "📐 Model Training"
                st.rerun()
        else:
            st.button("➡️ Go to Model Training (set target first)", disabled=True, use_container_width=True)

def training_page():
    st.markdown('<h2 class="sub-header">📐 Automated Model Training with FLAML</h2>', unsafe_allow_html=True)

    if not flaml_available:
        st.error("⚠️ FLAML is not installed. Please install it with `pip install flaml[automl]` to use this feature.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()
        return

    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload data and set target column first.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()
        return

    df = st.session_state.data
    target_col = st.session_state.target_column
    problem_type = st.session_state.problem_type

    st.markdown(f"""
    <div class="card">
    <h4>Training Configuration</h4>
    <ul>
        <li><strong>Problem Type:</strong> {problem_type}</li>
        <li><strong>Target Column:</strong> {target_col}</li>
        <li><strong>Dataset Shape:</strong> {df.shape}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if "train_time" not in st.session_state:
        st.session_state.train_time = 10
    if "training_mode" not in st.session_state:
        st.session_state.training_mode = "Balanced"

    col_mode, _ = st.columns([1, 2])
    with col_mode:
        mode = st.selectbox(
            "Training Mode Preset",
            ["Fast", "Balanced", "Accurate"],
            index=["Fast", "Balanced", "Accurate"].index(st.session_state.training_mode),
            help="Fast: short time; Accurate: longer time"
        )
    if mode != st.session_state.training_mode:
        st.session_state.training_mode = mode
        if mode == "Fast":
            st.session_state.train_time = 2
        elif mode == "Balanced":
            st.session_state.train_time = 10
        else:  # Accurate
            st.session_state.train_time = 30

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    with col2:
        time_budget_mins = st.slider("Time Budget (minutes)", 1, 60, value=st.session_state.train_time, key="train_time")
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)

    if st.button("🚀 Start Automated Training", type="primary", use_container_width=True):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        st.session_state.test_data = {'X_test': X_test, 'y_test': y_test}

        with st.spinner("🧠 FLAML is searching for the best model. This may take several minutes..."):
            try:
                task = 'classification' if problem_type == 'Classification' else 'regression'
                metric = 'accuracy' if task == 'classification' else 'r2'
                estimator_list = (
                    ["lgbm", "rf", "lrl1"] if task == "classification" else ["lgbm", "rf", "lrl2"]
                )

                automl = AutoML()
                automl.fit(
                    X_train, y_train,
                    task=task,
                    time_budget=time_budget_mins * 60,
                    metric=metric,
                    eval_method='holdout',
                    split_ratio=0.2,
                    estimator_list=estimator_list,
                    n_jobs=-1,
                    log_file_name='flaml.log',
                    verbose=0
                )

                with st.expander("📊 Training Results (click to expand)", expanded=True):
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown("#### 🏆 Best Model")
                        st.code(str(automl.model), language='python')
                    with col_res2:
                        st.markdown("#### ⚙️ Best Hyperparameters")
                        st.json(automl.best_config)

                    y_pred = automl.predict(X_test)
                    score = automl.score(X_test, y_test)
                    st.markdown(f"#### 📈 Test Score ({metric}): **{score:.4f}**")

                st.session_state.model = automl
                st.session_state.predictions = y_pred
                st.session_state.training_complete = True

                st.success("🎉 Model training completed successfully!")
                st.session_state.app_page = "📈 Model Evaluation"
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

def evaluation_page():
    st.markdown('<h2 class="sub-header">📈 Model Performance Evaluation</h2>', unsafe_allow_html=True)

    if not st.session_state.training_complete or st.session_state.model is None:
        st.warning("⚠️ No trained model found. Please go to 'Model Training' and train a model first.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "📐 Model Training"
            st.rerun()
        return

    if st.session_state.predictions is None or st.session_state.test_data is None:
        st.error("❌ Model predictions or test data are missing. Please retrain the model.")
        if st.button("Retrain Model"):
            st.session_state.app_page = "📐 Model Training"
            st.rerun()
        return

    model = st.session_state.model
    predictions = st.session_state.predictions
    test_data = st.session_state.test_data
    y_test = test_data['y_test']
    problem_type = st.session_state.problem_type

    try:
        y_test = np.asarray(y_test).ravel()
        predictions = np.asarray(predictions).ravel()

        if problem_type == "Classification":
            y_test = y_test.astype(str)
            predictions = predictions.astype(str)

            valid_mask = ~pd.isna(y_test) & ~pd.isna(predictions)
            if not np.all(valid_mask):
                st.warning(f"Detected {np.sum(~valid_mask)} invalid values and removed them before evaluation.")
                y_test = y_test[valid_mask]
                predictions = predictions[valid_mask]

            if len(y_test) == 0:
                st.error("No valid samples available for evaluation.")
                return

            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
            rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col3.metric("Recall", f"{rec:.4f}")
            col4.metric("F1-Score", f"{f1:.4f}")

            st.markdown("### 🎯 Confusion Matrix")
            try:
                cm = confusion_matrix(y_test, predictions)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to generate confusion matrix: {e}")

            st.markdown("### 📝 Detailed Classification Report")
            try:
                report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate classification report: {e}")

        else:  # Regression
            y_test = y_test.astype(float)
            predictions = predictions.astype(float)

            valid_mask = np.isfinite(y_test) & np.isfinite(predictions)
            if not np.all(valid_mask):
                st.warning(f"Detected {np.sum(~valid_mask)} invalid values and removed them before evaluation.")
                y_test = y_test[valid_mask]
                predictions = predictions[valid_mask]

            if len(y_test) == 0:
                st.error("No valid samples available for evaluation.")
                return

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.4f}")
            col2.metric("MSE", f"{mse:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")
            col4.metric("R² Score", f"{r2:.4f}")

            st.markdown("### 📈 Actual vs Predicted")
            try:
                fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'},
                                 title='Actual vs Predicted Values')
                max_val = max(max(y_test), max(predictions))
                min_val = min(min(y_test), min(predictions))
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                         mode='lines', name='Perfect Prediction',
                                         line=dict(color='red', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate actual vs predicted plot: {e}")

            st.markdown("### 📉 Residual Plot")
            try:
                residuals = y_test - predictions
                fig = px.scatter(x=predictions, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'},
                                 title='Residual Plot')
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate residual plot: {e}")

        st.markdown("### 🏆 Best Model Found by FLAML")
        if model is not None:
            st.markdown(f"**Model Object:** {model.model}")
            st.markdown(f"**Best Configuration:** {model.best_config}")

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.training_complete:
                if st.button("➡️ Go to Export Results", type="primary", use_container_width=True):
                    st.session_state.app_page = "💾 Export Results"
                    st.rerun()
            else:
                st.button("➡️ Go to Export Results (train model first)", disabled=True, use_container_width=True)

    except Exception as e:
        st.error(f"Unknown error occurred during evaluation: {str(e)}")
        st.info("Please try retraining the model or check the data format.")

def export_page():
    st.markdown('<h2 class="sub-header">💾 Export Model and Results</h2>', unsafe_allow_html=True)
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first to export results.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "📐 Model Training"
            st.rerun()
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Model Information")
        if st.button("Show Model Details"):
            st.write("**Best Model:**", st.session_state.model.model)
            st.write("**Best Config:**", st.session_state.model.best_config)
    with col2:
        st.markdown("#### 📊 Model Report")
        if st.button("Generate Model Report"):
            if st.session_state.data is not None:
                dataset_shape = st.session_state.data.shape
                feature_count = len(st.session_state.data.columns) - 1
            else:
                dataset_shape = "N/A"
                feature_count = "N/A"

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
- Best Model: {st.session_state.model.model}
- Best Hyperparameters: {st.session_state.model.best_config}
- Training Completed: {st.session_state.training_complete}

## Notes
This model was generated using FLAML AutoML through the No-Code ML Platform.
"""
            st.code(report_content, language='markdown')
            pdf_bytes = text_to_simple_pdf_bytes(report_content, title="ML Model Report")
            st.download_button(
                "📥 Download Report (PDF)",
                data=pdf_bytes,
                file_name="ml_model_report.pdf",
                mime="application/pdf"
            )
            st.download_button(
                "📥 Download Report (Markdown)",
                data=report_content,
                file_name="ml_model_report.md",
                mime="text/markdown"
            )

    st.markdown("### 📋 Session Information")
    session_info = {
        "Data Loaded": st.session_state.data is not None,
        "Target Column": st.session_state.target_column,
        "Problem Type": st.session_state.problem_type,
        "Model Trained": st.session_state.training_complete,
        "Predictions Made": st.session_state.predictions is not None,
        "Test Data Available": st.session_state.test_data is not None
    }
    session_df = pd.DataFrame.from_dict(session_info, orient='index', columns=['Status'])
    st.dataframe(session_df, use_container_width=True)

    st.markdown("### 🔄 Reset Platform")
    st.warning("This will clear all data and models from the current session.")
    if st.button("🔄 Reset All Data", type="secondary"):
        keys = ["data", "target_column", "problem_type", "model", "predictions", "test_data", "training_complete",
                "cleaned_data"]
        for key in keys:
            if key in st.session_state:
                st.session_state[key] = None
        st.rerun()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Start Over (Back to Data Upload)", type="secondary", use_container_width=True):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()

# ---------- Dashboard ----------
def dashboard_page():
    set_bg_image_local("purple.jpg")

    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background:#ffffe0 !important;
    }
    section[data-testid="stSidebar"] .st-emotion-cache-1wrcr25, 
    section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1 style='color: white;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
        st.markdown("### Sequential Steps")
        app_page_options = [
            "📁 Data Upload",
            "🧹 Data Cleaning",
            "🔍 Exploratory Data Analysis",
            "📐 Model Training",
            "📈 Model Evaluation",
            "💾 Export Results"
        ]
        # Safe index selection
        if st.session_state.app_page in app_page_options:
            default_index = app_page_options.index(st.session_state.app_page)
        else:
            default_index = 0
            st.session_state.app_page = app_page_options[0]  # sync with valid value

        selected = st.radio("Select a step:", app_page_options, index=default_index)
        st.session_state.app_page = selected

        st.markdown("---")
        st.markdown("### Platform Info")
        st.info("""
        This platform enables:
        - CSV data upload
        - Data cleaning
        - Automated EDA
        - AutoML with FLAML
        - Model evaluation
        - Export model report
        """)
        if not flaml_available:
            st.error("⚠️ FLAML not installed. Install with: `pip install flaml[automl]`")
            st.code("pip install flaml[automl]", language="bash")

        if st.button("👋🏻 Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            keys = ["data", "target_column", "problem_type", "model", "predictions", "test_data", "training_complete",
                    "cleaned_data"]
            for key in keys:
                if key in st.session_state:
                    st.session_state[key] = None
            go_to("front")
            st.rerun()

    # Main content area
    if st.session_state.app_page == "📁 Data Upload":
        upload_page()
    elif st.session_state.app_page == "🧹 Data Cleaning":
        cleaning_page()
    elif st.session_state.app_page == "🔍 Exploratory Data Analysis":
        eda_page()
    elif st.session_state.app_page == "📐 Model Training":
        training_page()
    elif st.session_state.app_page == "📈 Model Evaluation":
        evaluation_page()
    elif st.session_state.app_page == "💾 Export Results":
        export_page()

# ---------- Main routing ----------
if st.session_state.page == "front":
    front_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "dashboard":
    if not st.session_state.logged_in:
        go_to("login")
        st.rerun()
    else:
        dashboard_page()