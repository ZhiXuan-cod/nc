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
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

from supabase import create_client
import scipy.stats as stats  # for outlier detection
import pickle  # for model serialization

# ---------- PyCaret imports ----------
try:
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare, predict_model as clf_predict, get_config, pull, save_model as pycaret_save_model
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, predict_model as reg_predict
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    st.warning("⚠️ PyCaret not installed. Install with 'pip install pycaret' to use AutoML.")

# ---------- Minimal PDF generator (no extra installs) ----------
def _pdf_escape(text: str) -> str:
    return text.encode('ascii', 'ignore').decode('ascii').replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

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

def set_bg_image_local(image_base_name):
    for ext in ['.jpg', '.png']:
        full_path = image_base_name + ext
        bin_str = get_base64_of_file(full_path)
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
            return
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
    set_bg_image_local("FrontPage")
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
    set_bg_image_local("login")
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
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None

# ---------- Helper function for cleaning ----------
def apply_cleaning(df, drop_duplicates, missing_option, outlier_option, encode_option, scale_option, cols_to_drop):
    cleaned = df.copy()

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    if missing_option != "None":
        if missing_option == "Drop rows with any missing":
            cleaned = cleaned.dropna()
        elif missing_option == "Drop columns with any missing":
            cleaned = cleaned.dropna(axis=1)
        elif missing_option == "Fill numeric with mean":
            num_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
        elif missing_option == "Fill numeric with median":
            num_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        elif missing_option == "Fill categorical with mode":
            cat_cols = cleaned.select_dtypes(include=['object']).columns
            for col in cat_cols:
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0] if not cleaned[col].mode().empty else "Unknown")

    if outlier_option != "None":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
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

    if encode_option != "None":
        cat_cols = cleaned.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            if encode_option == "Label Encoding":
                for col in cat_cols:
                    le = LabelEncoder()
                    cleaned[col] = le.fit_transform(cleaned[col].astype(str))
            elif encode_option == "One-Hot Encoding":
                cleaned = pd.get_dummies(cleaned, columns=cat_cols, drop_first=True)

    if scale_option != "None":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            if scale_option == "Standardization (z-score)":
                scaler = StandardScaler()
                cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
            elif scale_option == "Normalization (min-max)":
                scaler = MinMaxScaler()
                cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])

    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop)

    return cleaned

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
            if uploaded_file.size > 200 * 1024 * 1024:
                st.warning("File size exceeds 200 MB. Large files may cause performance issues. Consider using a subset.")
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
    _, col2, _ = st.columns([1, 2, 1])
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
        cols_to_drop = st.multiselect("Select columns to drop",
                                      [c for c in original_df.columns if c != st.session_state.target_column])

        if st.button("🔍 Preview Cleaning", type="secondary"):
            cleaned = apply_cleaning(original_df, drop_duplicates, missing_option, outlier_option,
                                     encode_option, scale_option, cols_to_drop)
            st.markdown("### Cleaned Data Preview")
            st.dataframe(cleaned.head())
            st.markdown(f"Final shape: {cleaned.shape}")
            st.session_state.cleaned_data = cleaned

    st.markdown("---")
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.session_state.cleaned_data is not None:
            if st.button("✅ Apply Cleaning and Continue", type="primary", use_container_width=True):
                cleaned = apply_cleaning(original_df, drop_duplicates, missing_option, outlier_option,
                                         encode_option, scale_option, cols_to_drop)
                st.session_state.data = cleaned
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
            # ✅ FIX: renamed from 'stats' to 'col_stats' to avoid overwriting scipy.stats
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
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.session_state.target_column is not None:
            if st.button("➡️ Go to Model Training", type="primary", use_container_width=True):
                st.session_state.app_page = "📐 Model Training"
                st.rerun()
        else:
            st.button("➡️ Go to Model Training (set target first)", disabled=True, use_container_width=True)

def training_page():
    st.markdown('<h2 class="sub-header">📐 Automated Model Training with PyCaret</h2>', unsafe_allow_html=True)

    if not PYCARET_AVAILABLE:
        st.error("⚠️ PyCaret is not installed. Please install it with `pip install pycaret` to use AutoML.")
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

    df = st.session_state.data.copy()
    target_col = st.session_state.target_column
    problem_type = st.session_state.problem_type

    if problem_type == "Classification" and df[target_col].nunique() > 20:
        st.warning(f"Target column has {df[target_col].nunique()} unique values. Classification may be slow or have low accuracy. Consider regression or reduce categories.")

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
            help="Fast: lightweight models only; Balanced: mix of models; Accurate: more models & deeper tuning"
        )
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
        df = df.sample(frac=sample_frac, random_state=random_state)
        st.info(f"Using {len(df)} rows after sampling (original: {st.session_state.data.shape[0]}).")

    if st.button("🚀 Start Automated Training", type="primary", use_container_width=True):
        with st.spinner(f"🧠 PyCaret is training {len(allowed_models)} models with {fold}-fold CV. This may take a few minutes..."):
            try:
                if problem_type == "Classification":
                    clf_setup(
                        data=df,
                        target=target_col,
                        train_size=1 - test_size,
                        session_id=random_state,
                        fold=fold,
                        n_jobs=1,
                        html=False,
                        verbose=False,
                        ignore_low_variance=False,
                        remove_multicollinearity=False,
                        log_experiment=False
                    )
                    best_model = clf_compare(
                        include=allowed_models,
                        n_select=1,
                        verbose=False,
                        sort='Accuracy'
                    )
                else:
                    reg_setup(
                        data=df,
                        target=target_col,
                        train_size=1 - test_size,
                        session_id=random_state,
                        fold=fold,
                        n_jobs=1,
                        html=False,
                        verbose=False,
                        ignore_low_variance=False,
                        remove_multicollinearity=False,
                        log_experiment=False
                    )
                    best_model = reg_compare(
                        include=allowed_models,
                        n_select=1,
                        verbose=False,
                        sort='R2'
                    )

                try:
                    X_test = get_config('X_test')
                    y_test = get_config('y_test')
                except Exception as e:
                    st.error(f"Failed to retrieve test data: {e}")
                    return

                if problem_type == "Classification":
                    pred_df = clf_predict(best_model, data=X_test)
                else:
                    pred_df = reg_predict(best_model, data=X_test)
                test_predictions = pred_df.iloc[:, -1]

                st.session_state.test_data = {'X_test': X_test, 'y_test': y_test}
                st.session_state.predictions = test_predictions.values
                st.session_state.model = best_model
                st.session_state.training_complete = True

                with st.expander("📊 Training Results (click to expand)", expanded=True):
                    st.markdown("#### 🏆 Best Model")
                    st.code(str(best_model), language='python')
                    if hasattr(best_model, 'feature_importances_'):
                        st.markdown("#### 🔍 Feature Importance (top 10)")
                        imp_df = pd.DataFrame({'feature': X_test.columns, 'importance': best_model.feature_importances_})
                        imp_df = imp_df.sort_values('importance', ascending=False).head(10)
                        st.dataframe(imp_df, use_container_width=True)
                    comparison_df = pull()
                    if comparison_df is not None:
                        st.markdown("#### 📊 Model Comparison (top 10)")
                        st.dataframe(comparison_df.head(10), use_container_width=True)

                st.success("🎉 Model training completed successfully!")
                st.session_state.app_page = "📈 Model Evaluation"
                st.rerun()

            except Exception as e:
                st.error(f"❌ Training failed: {type(e).__name__}: {str(e)}")
                print(f"Training error: {type(e).__name__}: {e}")

# ---------- Helper to generate report text ----------
def generate_report_text(problem_type, metrics, model, target_col, dataset_shape=None):
    lines = []
    lines.append("# Machine Learning Model Evaluation Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Problem Type: {problem_type}")
    lines.append(f"Target Column: {target_col}")
    if dataset_shape:
        lines.append(f"Dataset Shape: {dataset_shape}")
    lines.append("")
    lines.append("## Performance Summary")
    for key, val in metrics.items():
        lines.append(f"- {key}: {val:.4f}" if isinstance(val, float) else f"- {key}: {val}")
    lines.append("")
    lines.append("## Best Model")
    lines.append(str(model))
    return "\n".join(lines)

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
    X_test = test_data['X_test']
    if isinstance(X_test, pd.DataFrame):
        feature_names = X_test.columns.tolist()
    else:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    try:
        y_test = np.asarray(y_test).ravel()
        predictions = np.asarray(predictions).ravel()

        with st.expander("🔍 Model Information", expanded=False):
            st.markdown("#### Best Model")
            st.code(str(model), language='python')
            try:
                if hasattr(model, 'feature_importances_'):
                    st.markdown("#### Feature Importance (all)")
                    imp_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
                    imp_df = imp_df.sort_values('importance', ascending=False)
                    st.dataframe(imp_df, use_container_width=True)
                elif hasattr(model, 'coef_'):
                    coef = model.coef_
                    if coef.ndim == 2:
                        imp_df = pd.DataFrame({'feature': feature_names, 'importance': np.abs(coef).mean(axis=0)})
                        imp_df = imp_df.sort_values('importance', ascending=False)
                        st.markdown("#### Mean Absolute Coefficients (Multi-class)")
                    else:
                        imp_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
                        imp_df = imp_df.sort_values('coefficient', ascending=False)
                        st.markdown("#### Coefficients (Top 20)")
                    st.dataframe(imp_df.head(20), use_container_width=True)
                else:
                    st.info("This model does not support feature importance display.")
            except Exception as e:
                st.warning(f"Feature importance display failed: {e}")

        if problem_type == "Classification":
            y_test_str = y_test.astype(str)
            predictions_str = predictions.astype(str)

            valid_mask = ~pd.isna(y_test_str) & ~pd.isna(predictions_str)
            if not np.all(valid_mask):
                st.warning(f"Detected {np.sum(~valid_mask)} invalid values and removed them before evaluation.")
                y_test_str = y_test_str[valid_mask]
                predictions_str = predictions_str[valid_mask]

            if len(y_test_str) == 0:
                st.error("No valid samples available for evaluation.")
                return

            acc = accuracy_score(y_test_str, predictions_str)
            prec = precision_score(y_test_str, predictions_str, average='weighted', zero_division=0)
            rec = recall_score(y_test_str, predictions_str, average='weighted', zero_division=0)
            f1 = f1_score(y_test_str, predictions_str, average='weighted', zero_division=0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col3.metric("Recall", f"{rec:.4f}")
            col4.metric("F1-Score", f"{f1:.4f}")

            st.markdown("### 📖 Interpretation")
            if acc > 0.9:
                st.success("✅ **Excellent accuracy** – the model correctly classifies >90% of cases.")
            elif acc > 0.7:
                st.info("ℹ️ **Good accuracy** – the model performs reasonably well, but there is room for improvement.")
            else:
                st.warning("⚠️ **Low accuracy** – consider collecting more data, feature engineering, or adjusting the model.")

            if prec > 0.8 and rec > 0.8:
                st.success("✅ **Balanced precision & recall** – the model is reliable both in positive predictions and capturing true positives.")
            elif prec < 0.5:
                st.warning("⚠️ **Low precision** – many false positives; may need to adjust classification threshold or use different metrics.")
            elif rec < 0.5:
                st.warning("⚠️ **Low recall** – many false negatives; the model misses a significant number of positive instances.")

            st.markdown("### 🎯 Confusion Matrix")
            try:
                cm = confusion_matrix(y_test_str, predictions_str)
                labels = sorted(set(y_test_str).union(set(predictions_str)))
                fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                                color_continuous_scale='Blues',
                                title="Confusion Matrix (Counts)")
                fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(fig, use_container_width=True)

                if len(labels) == 2:
                    tn, fp, fn, tp = cm.ravel()
                    st.markdown(f"""
                    - **True Negatives (TN):** {tn} – correctly predicted negative class.
                    - **False Positives (FP):** {fp} – incorrectly predicted positive (Type I error).
                    - **False Negatives (FN):** {fn} – missed positives (Type II error).
                    - **True Positives (TP):** {tp} – correctly predicted positive.
                    """)
                else:
                    st.info("Multiclass confusion matrix – examine the diagonal for correct predictions.")
            except Exception as e:
                st.error(f"Failed to generate confusion matrix: {e}")

            if hasattr(model, 'predict_proba') and len(np.unique(y_test_str)) == 2:
                try:
                    if hasattr(model, 'classes_'):
                        pos_class = model.classes_[1]
                    else:
                        le = LabelEncoder()
                        le.fit(y_test_str)
                        pos_class = le.classes_[1]
                    y_proba = model.predict_proba(X_test)[:, 1]
                    y_test_num = (y_test_str == pos_class).astype(int)

                    fpr, tpr, _ = roc_curve(y_test_num, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})'))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                    fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                                            title='ROC Curve')
                    st.plotly_chart(fig_roc, use_container_width=True)

                    precisions, recalls, _ = precision_recall_curve(y_test_num, y_proba)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name='PR Curve'))
                    fig_pr.update_layout(xaxis_title='Recall', yaxis_title='Precision',
                                            title='Precision-Recall Curve')
                    st.plotly_chart(fig_pr, use_container_width=True)
                except Exception as e:
                    st.info(f"Could not compute ROC/PR curves: {e}")

            if hasattr(model, 'predict_proba') and len(np.unique(y_test_str)) == 2:
                st.markdown("### ⚙️ Decision Threshold Tuning & Cost Simulation")
                st.write("Adjust the classification threshold to optimize for your business needs.")
                y_proba = model.predict_proba(X_test)[:, 1]
                threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
                y_pred_adj = (y_proba >= threshold).astype(int)
                if hasattr(model, 'classes_'):
                    le = LabelEncoder()
                    le.fit(model.classes_)
                else:
                    le = LabelEncoder()
                    le.fit(y_test_str)
                y_pred_adj_labels = le.inverse_transform(y_pred_adj)

                tn, fp, fn, tp = confusion_matrix(y_test_str, y_pred_adj_labels).ravel()
                st.markdown("**Simulate business costs:**")
                col1, col2 = st.columns(2)
                with col1:
                    cost_fp = st.number_input("Cost per False Positive ($)", min_value=0.0, value=10.0, step=1.0)
                with col2:
                    cost_fn = st.number_input("Cost per False Negative ($)", min_value=0.0, value=100.0, step=1.0)
                total_cost = fp * cost_fp + fn * cost_fn
                st.metric("Total Simulated Cost", f"${total_cost:,.2f}")

                adj_acc = accuracy_score(y_test_str, y_pred_adj_labels)
                adj_prec = precision_score(y_test_str, y_pred_adj_labels, average='binary', pos_label=le.classes_[1])
                adj_rec = recall_score(y_test_str, y_pred_adj_labels, average='binary', pos_label=le.classes_[1])
                st.write(f"At threshold {threshold:.2f}: Accuracy={adj_acc:.4f}, Precision={adj_prec:.4f}, Recall={adj_rec:.4f}")

            st.markdown("### 📝 Detailed Classification Report")
            try:
                report = classification_report(y_test_str, predictions_str, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate classification report: {e}")

            st.markdown("---")
            metrics = {"Accuracy": acc, "Weighted Precision": prec, "Weighted Recall": rec, "Weighted F1": f1}
            report_text = generate_report_text("Classification", metrics, model, st.session_state.target_column)
            report_text += "\n\n### Confusion Matrix\n" + str(confusion_matrix(y_test_str, predictions_str))
            report_text += "\n\n### Classification Report\n" + classification_report(y_test_str, predictions_str, zero_division=0)

            col1, col2 = st.columns([1, 2])
            with col1:
                pdf_bytes = text_to_simple_pdf_bytes(report_text, title="ML Evaluation Report")
                st.download_button(
                    label="📥 Download Full Evaluation Report (PDF)",
                    data=pdf_bytes,
                    file_name="ml_evaluation_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="cls_pdf"
                )
            with col2:
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=report_text,
                    file_name="ml_evaluation_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="cls_md"
                )

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

            st.markdown("### 📖 Interpretation")
            if r2 > 0.8:
                st.success("✅ **Excellent R²** – the model explains >80% of the variance.")
            elif r2 > 0.5:
                st.info("ℹ️ **Moderate R²** – the model explains about half the variance.")
            else:
                st.warning("⚠️ **Low R²** – the model does not capture much variance; consider more features or different models.")

            st.markdown(f"**Mean Absolute Error (MAE):** On average, predictions are off by {mae:.2f} units.")
            st.markdown(f"**Root Mean Squared Error (RMSE):** Heavily penalizes large errors; current value: {rmse:.2f}.")

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

                residuals_mean = np.mean(residuals)
                residuals_std = np.std(residuals)
                st.markdown(f"- **Mean of residuals:** {residuals_mean:.4f} (should be close to zero).")
                st.markdown(f"- **Standard deviation:** {residuals_std:.4f}.")
                if abs(residuals_mean) > 0.1 * rmse:
                    st.warning("⚠️ Residuals have a non-zero mean – there might be systematic bias in predictions.")
                else:
                    st.success("✅ Residuals are roughly centered around zero, indicating no systematic bias.")
            except Exception as e:
                st.error(f"Failed to generate residual plot: {e}")

            st.markdown("---")
            metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}
            report_text = generate_report_text("Regression", metrics, model, st.session_state.target_column)

            col1, col2 = st.columns([1, 2])
            with col1:
                pdf_bytes = text_to_simple_pdf_bytes(report_text, title="ML Evaluation Report")
                st.download_button(
                    label="📥 Download Full Evaluation Report (PDF)",
                    data=pdf_bytes,
                    file_name="ml_evaluation_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="reg_pdf"
                )
            with col2:
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=report_text,
                    file_name="ml_evaluation_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="reg_md"
                )

        st.markdown("### 🏆 Best Model Found by PyCaret")
        if model is not None:
            st.code(str(model), language='python')
            try:
                if hasattr(model, 'feature_importances_'):
                    st.markdown("#### Feature Importance (Top 20)")
                    imp_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
                    imp_df = imp_df.sort_values('importance', ascending=False).head(20)
                    fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h', title='Feature Importance')
                    st.plotly_chart(fig_imp, use_container_width=True)
                elif hasattr(model, 'coef_'):
                    coef = model.coef_
                    if coef.ndim == 2:
                        imp_df = pd.DataFrame({'feature': feature_names, 'importance': np.abs(coef).mean(axis=0)})
                    else:
                        imp_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
                    imp_df = imp_df.sort_values('importance', ascending=False).head(20)
                    fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h', title='Feature Coefficients')
                    st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot feature importance: {e}")

        st.markdown("---")
        _, col2, _ = st.columns([1, 2, 1])
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
            st.write("**Best Model:**", st.session_state.model)
        if st.button("💾 Download Model (pickle)"):
            model_bytes = pickle.dumps(st.session_state.model)
            st.download_button(
                label="Click to download model",
                data=model_bytes,
                file_name="ml_model.pkl",
                mime="application/octet-stream",
                key="model_download"
            )
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
- Best Model: {st.session_state.model}
- Training Completed: {st.session_state.training_complete}

## Notes
This model was generated using PyCaret AutoML through the No-Code ML Platform.
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
                "cleaned_data", "label_encoder"]
        for key in keys:
            if key in st.session_state:
                st.session_state[key] = None
        st.session_state.app_page = "📁 Data Upload"
        st.rerun()

    st.markdown("---")
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Start Over (Back to Data Upload)", type="secondary", use_container_width=True):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()

# ---------- Dashboard ----------
def dashboard_page():
    set_bg_image_local("purple")

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

    st.markdown(f"<h1 style='color: black;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)

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
        if st.session_state.app_page in app_page_options:
            default_index = app_page_options.index(st.session_state.app_page)
        else:
            default_index = 0
            st.session_state.app_page = app_page_options[0]

        selected = st.radio("Select a step:", app_page_options, index=default_index)
        st.session_state.app_page = selected

        st.markdown("---")
        st.markdown("### Platform Info")
        st.info("""
        This platform enables:
        - CSV data upload
        - Data cleaning
        - Automated EDA
        - AutoML with PyCaret
        - Model evaluation with interpretability
        - Export model report
        """)
        if not PYCARET_AVAILABLE:
            st.error("⚠️ PyCaret not installed. Install with: `pip install pycaret`")
            st.code("pip install pycaret", language="bash")

        if st.button("👋🏻 Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            keys = ["data", "target_column", "problem_type", "model", "predictions", "test_data", "training_complete",
                    "cleaned_data", "label_encoder"]
            for key in keys:
                if key in st.session_state:
                    st.session_state[key] = None
            go_to("front")
            st.rerun()

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