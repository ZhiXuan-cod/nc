import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

from supabase import create_client

# ---------- TPOT availability ----------
try:
    from tpot import TPOTClassifier, TPOTRegressor
    tpot_available = True
except ImportError:
    tpot_available = False
    st.error(f"TPOT is not installed: {e}")

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
if "imputer_num" not in st.session_state:
    st.session_state.imputer_num = None
if "imputer_cat" not in st.session_state:
    st.session_state.imputer_cat = None
if "num_cols" not in st.session_state:
    st.session_state.num_cols = None
if "cat_cols" not in st.session_state:
    st.session_state.cat_cols = None

# ---------- 以下为Dashboard子页面 ----------
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
                st.session_state.app_page = "🔍 Exploratory Analysis"
                st.rerun()

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

def training_page():
    st.markdown('<h2 class="sub-header">📐 Automated Model Training with TPOT</h2>', unsafe_allow_html=True)
    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload data and set target column first.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()
        return
    if not tpot_available:
        st.error("TPOT is not installed. Please install it to use AutoML features.")
        st.code("pip install tpot", language="bash")
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

    # ---------- 初始化会话状态变量（用于绑定滑块）----------
    if "train_gens" not in st.session_state:
        st.session_state.train_gens = 10
    if "train_pop" not in st.session_state:
        st.session_state.train_pop = 50
    if "train_cv" not in st.session_state:
        st.session_state.train_cv = 5
    if "train_time" not in st.session_state:
        st.session_state.train_time = 10
    if "training_mode" not in st.session_state:
        st.session_state.training_mode = "Balanced"

    # ---------- 训练模式选择（预设值）----------
    col_mode, _ = st.columns([1, 2])
    with col_mode:
        mode = st.selectbox(
            "Training Mode Preset",
            ["Fast", "Balanced", "Accurate"],
            index=["Fast", "Balanced", "Accurate"].index(st.session_state.training_mode),
            help="快速：降低参数以减少训练时间；精确：增加参数以追求更高性能"
        )
    # 当模式改变时，更新对应的会话状态变量
    if mode != st.session_state.training_mode:
        st.session_state.training_mode = mode
        if mode == "Fast":
            st.session_state.train_gens = 5
            st.session_state.train_pop = 20
            st.session_state.train_cv = 3
            st.session_state.train_time = 5
        elif mode == "Balanced":
            st.session_state.train_gens = 10
            st.session_state.train_pop = 50
            st.session_state.train_cv = 5
            st.session_state.train_time = 10
        else:  # Accurate
            st.session_state.train_gens = 50
            st.session_state.train_pop = 100
            st.session_state.train_cv = 10
            st.session_state.train_time = 30

    # ---------- 参数滑块（绑定会话状态）----------
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        generations = st.slider("Generations", 5, 50, value=st.session_state.train_gens, key="train_gens")
    with col2:
        population_size = st.slider("Population Size", 10, 100, value=st.session_state.train_pop, key="train_pop")
        cv_folds = st.slider("CV Folds", 2, 10, value=st.session_state.train_cv, key="train_cv")
    with col3:
        max_time_mins = st.slider("Max Time (minutes)", 1, 60, value=st.session_state.train_time, key="train_time")
        random_state = st.number_input("Random State", 0, 100, 42)

    handle_missing = st.selectbox("Handle Missing Values", ["auto", "impute", "drop"])

    if st.button("🚀 Start Automated Training", type="primary", use_container_width=True):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.isnull().any().any() or y.isnull().any():
            if handle_missing == "drop":
                valid_idx = X.dropna().index.intersection(y.dropna().index)
                X = X.loc[valid_idx]
                y = y.loc[valid_idx]
                st.info(f"Dropped rows with missing values. Remaining: {len(X)} rows.")

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        st.session_state.num_cols = num_cols
        st.session_state.cat_cols = cat_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        st.session_state.test_data = {'X_test': X_test, 'y_test': y_test}

        imputer_num = None
        imputer_cat = None
        if handle_missing in ["auto", "impute"] and (X_train.isnull().any().any() or y_train.isnull().any()):
            if num_cols:
                imputer_num = SimpleImputer(strategy='mean')
                X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
                X_test[num_cols] = imputer_num.transform(X_test[num_cols])
            if cat_cols:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
                X_test[cat_cols] = imputer_cat.transform(X_test[cat_cols])
            st.info("Missing values imputed (mean for numerical, mode for categorical).")

        st.session_state.imputer_num = imputer_num
        st.session_state.imputer_cat = imputer_cat

        cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

        with st.spinner("🧠 TPOT is searching for the best pipeline. This may take several minutes..."):
            try:
                if problem_type == "Classification":
                    tpot = TPOTClassifier(
                        generations=generations,
                        population_size=population_size,
                        cv=cv_folds,
                        random_state=random_state,
                        verbosity=2,
                        n_jobs=-1,
                        max_time_mins=max_time_mins,
                        categorical_features=cat_indices if cat_indices else None
                    )
                else:
                    tpot = TPOTRegressor(
                        generations=generations,
                        population_size=population_size,
                        cv=cv_folds,
                        random_state=random_state,
                        verbosity=2,
                        n_jobs=-1,
                        max_time_mins=max_time_mins,
                        categorical_features=cat_indices if cat_indices else None
                    )

                tpot.fit(X_train, y_train)
                st.session_state.model = tpot
                st.session_state.predictions = tpot.predict(X_test)
                st.session_state.training_complete = True

                st.success("🎉 Model training completed successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

def evaluation_page():
    st.markdown('<h2 class="sub-header">📈 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    if not st.session_state.training_complete or st.session_state.model is None:
        st.warning("⚠️ Please train a model first from the 'Model Training' page.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "📐 Model Training"
            st.rerun()
        return

    model = st.session_state.model
    predictions = st.session_state.predictions
    test_data = st.session_state.test_data
    y_test = test_data['y_test']
    problem_type = st.session_state.problem_type

    if problem_type == "Classification":
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
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        st.markdown("### 📝 Detailed Classification Report")
        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    else:
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
        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'},
            title='Actual vs Predicted Values')
        max_val = max(max(y_test), max(predictions))
        min_val = min(min(y_test), min(predictions))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📉 Residual Plot")
        residuals = y_test - predictions
        fig = px.scatter(x=predictions, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'},
                        title='Residual Plot')
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 Best Pipeline Found by TPOT")
    if model is not None:
        st.code(model.fitted_pipeline_, language='python')
        pipeline_code = model.export()
        st.download_button("📥 Download Pipeline Code", data=pipeline_code,
                        file_name="best_pipeline.py", mime="text/python")

def prediction_page():
    st.markdown('<h2 class="sub-header">🔮 Make Predictions with Trained Model</h2>', unsafe_allow_html=True)
    if not st.session_state.training_complete or st.session_state.model is None:
        st.warning("⚠️ Please train a model first from the 'Model Training' page.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "📐 Model Training"
            st.rerun()
        return

    model = st.session_state.model
    method = st.radio("Select prediction method:",
                    ["📤 Upload New Data", "✍️ Manual Input", "📊 Use Test Data"])

    if method == "📤 Upload New Data":
        st.markdown("### 📤 Upload New Data for Prediction")
        new_file = st.file_uploader("Upload new CSV file for predictions", type=['csv'], key="pred_file")
        if new_file is not None:
            try:
                new_df = pd.read_csv(new_file)
                original_cols = st.session_state.data.drop(columns=[st.session_state.target_column]).columns.tolist()
                missing_cols = set(original_cols) - set(new_df.columns)
                if missing_cols:
                    st.warning(f"⚠️ Missing columns: {missing_cols}. They will be filled with 0.")
                new_df = new_df.reindex(columns=original_cols, fill_value=0)

                if st.session_state.imputer_num is not None and st.session_state.num_cols:
                    new_df[st.session_state.num_cols] = st.session_state.imputer_num.transform(new_df[st.session_state.num_cols])
                if st.session_state.imputer_cat is not None and st.session_state.cat_cols:
                    new_df[st.session_state.cat_cols] = st.session_state.imputer_cat.transform(new_df[st.session_state.cat_cols])

                st.markdown("### 📋 Data Preview")
                st.dataframe(new_df.head(), use_container_width=True)

                if st.button("🔮 Make Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        preds = model.predict(new_df)
                        results_df = new_df.copy()
                        results_df['Predictions'] = preds
                        st.success(f"✅ Predictions complete for {len(preds)} samples!")
                        st.dataframe(results_df, use_container_width=True)
                        csv = results_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions</a>'
                        st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    elif method == "✍️ Manual Input":
        st.markdown("### ✍️ Enter Values Manually")
        feature_cols = st.session_state.data.drop(columns=[st.session_state.target_column]).columns.tolist()
        input_data = {}
        cols = st.columns(3)
        for i, col_name in enumerate(feature_cols):
            with cols[i % 3]:
                if col_name in st.session_state.num_cols:
                    min_val = float(st.session_state.data[col_name].min())
                    max_val = float(st.session_state.data[col_name].max())
                    mean_val = float(st.session_state.data[col_name].mean())
                    input_data[col_name] = st.number_input(col_name, min_value=min_val, max_value=max_val, value=mean_val)
                else:
                    unique_vals = st.session_state.data[col_name].unique()[:10]
                    input_data[col_name] = st.selectbox(col_name, unique_vals)
        if st.button("🔮 Predict", type="primary"):
            input_df = pd.DataFrame([input_data])
            if st.session_state.imputer_num is not None and st.session_state.num_cols:
                input_df[st.session_state.num_cols] = st.session_state.imputer_num.transform(input_df[st.session_state.num_cols])
            if st.session_state.imputer_cat is not None and st.session_state.cat_cols:
                input_df[st.session_state.cat_cols] = st.session_state.imputer_cat.transform(input_df[st.session_state.cat_cols])
            pred = model.predict(input_df)[0]
            st.markdown(f"""
            <div class="success-box">
            <h3>Predicted {st.session_state.target_column}: {pred}</h3>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("### 📊 Predictions on Test Data")
        if st.session_state.test_data is not None:
            X_test = st.session_state.test_data['X_test']
            y_test = st.session_state.test_data['y_test']
            preds = model.predict(X_test)
            comp_df = X_test.copy()
            comp_df['Actual'] = y_test.values
            comp_df['Predicted'] = preds
            st.dataframe(comp_df.head(20), use_container_width=True)
            csv = comp_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="test_predictions.csv">📥 Download Test Predictions</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No test data available. Please train a model first.")

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
        st.markdown("#### 🐍 Python Code")
        if st.button("Generate Pipeline Code"):
            pipeline_code = st.session_state.model.export()
            st.code(pipeline_code, language='python')
            st.download_button("📥 Download Pipeline", data=pipeline_code,
                            file_name="tpot_pipeline.py", mime="text/python")
    with col2:
        st.markdown("#### 📊 Model Report")
        if st.button("Generate Model Report"):
            report_content = f"""
# Machine Learning Model Report

## Project Information
- Platform: No-Code ML Platform
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Problem Type: {st.session_state.problem_type}
- Target Column: {st.session_state.target_column}

## Dataset Information
- Original Shape: {st.session_state.data.shape if st.session_state.data else 'N/A'}
- Features: {len(st.session_state.data.columns) - 1 if st.session_state.data else 'N/A'}

## Model Information
- Best Pipeline: {st.session_state.model.fitted_pipeline_ if st.session_state.model else 'N/A'}
- Training Completed: {st.session_state.training_complete}

## Notes
This model was generated using TPOT AutoML through the No-Code ML Platform.
"""
            st.code(report_content, language='markdown')
            st.download_button("📥 Download Report", data=report_content,
                            file_name="ml_model_report.md", mime="text/markdown")

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
                "imputer_num", "imputer_cat", "num_cols", "cat_cols"]
        for key in keys:
            if key in st.session_state:
                st.session_state[key] = None
        st.rerun()

# ---------- 仪表盘 Dashboard ----------
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
        st.markdown("### Sequential Steps")
        app_page_options = [
            "📁 Data Upload",
            "🔍 Exploratory Analysis",
            "📐 Model Training",
            "📈 Model Evaluation",
            "🔮 Make Predictions",
            "💾 Export Results"
        ]
        selected = st.radio("Select a step:", app_page_options, index=app_page_options.index(st.session_state.app_page))
        st.session_state.app_page = selected

        st.markdown("---")
        st.markdown("### Platform Info")
        st.info("""
        This platform enables:
        - CSV data upload
        - Automated EDA
        - AutoML with TPOT
        - Model evaluation
        - No-code predictions
        """)
        if not tpot_available:
            st.error("⚠️ TPOT not installed. Install with: `pip install tpot`")
            st.code("pip install tpot", language="bash")

        if st.button("👋🏻 Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            keys = ["data", "target_column", "problem_type", "model", "predictions", "test_data", "training_complete",
                    "imputer_num", "imputer_cat", "num_cols", "cat_cols"]
            for key in keys:
                if key in st.session_state:
                    st.session_state[key] = None
            go_to("front")
            st.rerun()

    # 主内容区域
    if st.session_state.app_page == "📁 Data Upload":
        upload_page()
    elif st.session_state.app_page == "🔍 Exploratory Analysis":
        eda_page()
    elif st.session_state.app_page == "📐 Model Training":
        training_page()
    elif st.session_state.app_page == "📈 Model Evaluation":
        evaluation_page()
    elif st.session_state.app_page == "🔮 Make Predictions":
        prediction_page()
    elif st.session_state.app_page == "💾 Export Results":
        export_page()

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