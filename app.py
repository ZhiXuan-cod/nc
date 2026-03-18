import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# TPOT & additional libraries
import tpot
from tpot import TPOTClassifier, TPOTRegressor
import pickle
from io import BytesIO, StringIO
import contextlib
import sys

from supabase import create_client

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Supabase 客户端 ----------
if "supabase" not in st.session_state:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        st.session_state.supabase = create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        st.session_state.supabase = None

# ---------- 背景图片（Base64嵌入）----------
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

# ---------- 用户数据存储 (Supabase) ----------
def register_user(email, password, name):
    """注册新用户到 Supabase"""
    if st.session_state.supabase is None:
        return False, "Supabase not connected"
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) > 0:
            return False, "Email already registered."
        data = {"email": email, "name": name, "password": password}
        st.session_state.supabase.table("users").insert(data).execute()
        return True, "Registration successful. Please log in."
    except Exception as e:
        return False, f"Registration failed: {e}"

def authenticate_user(email, password):
    """验证用户凭据"""
    if st.session_state.supabase is None:
        return False, None
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) == 0:
            return False, None
        user = response.data[0]
        if user["password"] == password:  # 建议生产环境使用密码哈希
            return True, user["name"]
        else:
            return False, None
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False, None

# ---------- 页面导航 ----------
if "page" not in st.session_state:
    st.session_state.page = "front"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

def go_to(page):
    st.session_state.page = page

# ---------- 全局CSS样式（按钮渐变、卡片等）----------
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
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* 全局按钮样式：蓝到紫渐变 */
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

# ---------- 首页 Front Page ----------
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
    .st-emotion-cache-ocqkz7 {
        display: flex;
        align-items: center;
        justify-content: center;
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
            <p>Accessible for Machine Learning without code.</p>
        </div>
        """
        st.markdown(right_html, unsafe_allow_html=True)
        if st.button("Get Started", key="get_started", use_container_width=True):
            go_to("login")
            st.rerun()

# ---------- 登录/注册页面（居中）----------
def login_page():
    set_bg_image_local("login.jpg")
    
    st.markdown("""
    <style>
    /* 使内容垂直居中 */
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
    .back-button-container button:active {
        transform: scale(0.98) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 三列布局，中间列放置表单实现水平居中
    col1, col2, col3 = st.columns([1, 2, 1])
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

# ---------- 初始化会话状态 ----------
if "data" not in st.session_state:
    st.session_state.data = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "model" not in st.session_state:          # Will store the trained TPOT object
    st.session_state.model = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "test_data" not in st.session_state:
    st.session_state.test_data = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "X_columns" not in st.session_state:      # Store feature column names
    st.session_state.X_columns = None

# ---------- Dashboard 页面 ----------
def dashboard_page():
    # 设置紫色背景
    set_bg_image_local("purple.jpg")
    
    # 侧边栏渐变样式
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background:#ffffe0 !important;
    }
    section[data-testid="stSidebar"] .css-1d391kg {
        background: transparent !important;
    }
    /* 侧边栏文字颜色设为白色 */
    section[data-testid="stSidebar"] .st-emotion-cache-1wrcr25, 
    section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1 style='color: white;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
        st.markdown("### ML Workflow")
        # Navigation
        step = st.sidebar.radio(
            "Go to",
            ["📁 Data Upload", "🔍 Data Overview & Target", "🤖 TPOT Training", "📊 Results & Predictions"],
            key="nav_step"
        )

        if st.button("👋🏻 Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            keys = ["data", "target_column", "problem_type", "model", "training_complete", "X_columns"]
            for key in keys:
                if key in st.session_state:
                    st.session_state[key] = None
            go_to("front")
            st.rerun()

    # ---------- Step 1: Data Upload ----------
    def data_upload_step():
        st.header("📁 Upload your dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.data = df
                # Reset downstream states
                st.session_state.training_complete = False
                st.session_state.model = None
                st.session_state.target_column = None
                st.success("File successfully loaded!")
                st.dataframe(df.head())
                st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            if st.session_state.data is not None:
                st.info("Using previously uploaded data.")
                st.dataframe(st.session_state.data.head())
            else:
                st.info("Please upload a file to begin.")

    # ---------- Step 2: Data Overview & Target Selection ----------
    def data_overview_step():
        if st.session_state.data is None:
            st.warning("Please upload data first.")
            return
        df = st.session_state.data
        st.subheader("🔍 Data Overview")
        st.write("First few rows:")
        st.dataframe(df.head())
        st.write("Data types:")
        st.write(df.dtypes)

        # Target column selection
        target = st.selectbox("Select the target column", df.columns.tolist(), index=None)
        if target:
            st.session_state.target_column = target
            # Simple problem type detection
            y = df[target]
            if y.dtype in ['object', 'category'] or y.nunique() < 10:
                default_type = "classification"
            else:
                default_type = "regression"
            problem_type = st.radio(
                "Problem type",
                ["classification", "regression"],
                index=0 if default_type == "classification" else 1
            )
            st.session_state.problem_type = problem_type
            st.session_state.X_columns = [col for col in df.columns if col != target]
            st.write(f"Features ({len(st.session_state.X_columns)}): {st.session_state.X_columns}")
            # Reset training because target changed
            st.session_state.training_complete = False
            st.session_state.model = None
        else:
            st.session_state.target_column = None

    # ---------- Step 3: TPOT Training ----------
    def tpot_training_step():
        if st.session_state.data is None or st.session_state.target_column is None:
            st.warning("Please upload data and select target column first.")
            return
        st.header("🤖 TPOT AutoML Training")
        st.write(f"Target: **{st.session_state.target_column}** ({st.session_state.problem_type})")
        st.write(f"Features: {len(st.session_state.X_columns)} columns")

        # TPOT parameters
        generations = st.number_input("Generations", min_value=1, max_value=100, value=5, step=1)
        population_size = st.number_input("Population size", min_value=1, max_value=100, value=10, step=1)
        cv = st.number_input("Cross-validation folds", min_value=2, max_value=10, value=5, step=1)
        
        if st.session_state.problem_type == "classification":
            scoring_options = ["accuracy", "f1", "roc_auc", "average_precision"]
        else:
            scoring_options = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
        scoring = st.selectbox("Scoring metric", scoring_options)
        
        verbosity = st.slider("Verbosity level", 0, 3, 2)
        random_state = st.number_input("Random state", value=42)

        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("TPOT is optimizing pipelines... This may take a while."):
                # Prepare data
                df = st.session_state.data
                target = st.session_state.target_column
                # Drop rows with missing target
                df_clean = df.dropna(subset=[target])
                y = df_clean[target]
                X = df_clean.drop(columns=[target])

                # Show categorical/numeric info
                cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                st.write(f"Categorical columns: {cat_cols}")
                st.write(f"Numeric columns: {num_cols}")

                # Instantiate TPOT
                if st.session_state.problem_type == "classification":
                    tpot = TPOTClassifier(
                        generations=generations,
                        population_size=population_size,
                        cv=cv,
                        scoring=scoring,
                        verbosity=verbosity,
                        random_state=random_state,
                        n_jobs=-1
                    )
                else:
                    tpot = TPOTRegressor(
                        generations=generations,
                        population_size=population_size,
                        cv=cv,
                        scoring=scoring,
                        verbosity=verbosity,
                        random_state=random_state,
                        n_jobs=-1
                    )

                # Capture stdout to display logs in an expander
                log_output = StringIO()
                old_stdout = sys.stdout
                sys.stdout = log_output

                try:
                    tpot.fit(X, y)
                except Exception as e:
                    st.error(f"TPOT training failed: {e}")
                    sys.stdout = old_stdout
                    return
                finally:
                    sys.stdout = old_stdout

                # Show logs
                with st.expander("Training Log", expanded=True):
                    st.text(log_output.getvalue())

                st.success("Training completed!")
                st.session_state.model = tpot
                st.session_state.training_complete = True

                # Display best pipeline and score
                st.subheader("Best Pipeline")
                st.code(str(tpot.fitted_pipeline_))

                # Show best CV score
                if hasattr(tpot, '_optimized_pipeline_score'):
                    st.metric("Best CV Score", f"{tpot._optimized_pipeline_score:.4f}")
                else:
                    st.metric("Best CV Score", "Not available")

    # ---------- Step 4: Results & Predictions ----------
    def results_step():
        if not st.session_state.training_complete or st.session_state.model is None:
            st.warning("Please train a model first.")
            return
        st.header("📊 Results and Predictions")

        tpot = st.session_state.model
        pipeline = tpot.fitted_pipeline_

        st.subheader("Best Pipeline")
        st.code(str(pipeline))

        # Download model
        st.subheader("Download Model")
        if pipeline:
            buf = BytesIO()
            pickle.dump(pipeline, buf)
            buf.seek(0)
            st.download_button(
                label="📥 Download Model (pickle)",
                data=buf,
                file_name="tpot_model.pkl",
                mime="application/octet-stream"
            )

        # Prediction on new data
        st.subheader("Make Predictions on New Data")
        pred_file = st.file_uploader("Upload new data (CSV) for prediction", type=["csv"], key="pred_upload")
        if pred_file is not None:
            try:
                new_df = pd.read_csv(pred_file)
                st.dataframe(new_df.head())
                # Ensure required columns exist
                X_columns = st.session_state.X_columns
                missing_cols = set(X_columns) - set(new_df.columns)
                if missing_cols:
                    st.error(f"Missing columns in new data: {missing_cols}")
                else:
                    X_new = new_df[X_columns]
                    predictions = pipeline.predict(X_new)
                    st.write("Predictions:")
                    st.write(predictions)
                    # Option for classification probabilities
                    if st.session_state.problem_type == "classification" and st.checkbox("Show probabilities"):
                        if hasattr(pipeline, "predict_proba"):
                            probs = pipeline.predict_proba(X_new)
                            st.write("Prediction probabilities:")
                            st.write(probs)
                        else:
                            st.info("This pipeline does not support probability prediction.")
                    # Download predictions
                    pred_df = pd.DataFrame({"Prediction": predictions})
                    csv = pred_df.to_csv(index=False)
                    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ---------- Route to the selected step ----------
    if step == "📁 Data Upload":
        data_upload_step()
    elif step == "🔍 Data Overview & Target":
        data_overview_step()
    elif step == "🤖 TPOT Training":
        tpot_training_step()
    elif step == "📊 Results & Predictions":
        results_step()

# ---------- 主路由 ----------
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