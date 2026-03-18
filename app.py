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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
import sys
import traceback

warnings.filterwarnings('ignore')

# ---------- 错误处理装饰器 ----------
def safe_run(func):
    """装饰器：捕获函数内的异常并显示在界面上"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"**Error in {func.__name__}:** {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    return wrapper

# ---------- PyCaret availability ----------
pycaret_available = False
try:
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare, predict_model, pull as clf_pull, save_model, plot_model
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull
    pycaret_available = True
except ImportError as e:
    st.error(f"PyCaret import failed: {e}. Please install with `pip install pycaret`")

# Try to import get_config (internal PyCaret helper)
try:
    from pycaret.internal.pycaret_experiment import get_config
    get_config_available = True
except ImportError:
    get_config_available = False

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- 模拟 Supabase 用户认证（如果没有配置 secrets）----------
# 为了简化，如果没有 Supabase，则使用内存字典模拟
if "users_db" not in st.session_state:
    st.session_state.users_db = {}  # 模拟数据库: email -> {"name": name, "password": password}

def register_user(email, password, name):
    """模拟注册"""
    if email in st.session_state.users_db:
        return False, "Email already registered."
    st.session_state.users_db[email] = {"name": name, "password": password}
    return True, "Registration successful. Please log in."

def authenticate_user(email, password):
    """模拟登录"""
    user = st.session_state.users_db.get(email)
    if user and user["password"] == password:
        return True, user["name"]
    return False, None

# ---------- 背景图片 ----------
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

# ---------- 页面导航 ----------
if "page" not in st.session_state:
    st.session_state.page = "front"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

def go_to(page):
    st.session_state.page = page

# ---------- 全局CSS样式 ----------
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; padding: 1rem; margin-bottom: 2rem; }
    .sub-header { font-size: 1.5rem; color: #3949AB; margin-top: 1.5rem; margin-bottom: 1rem; }
    .card { background-color: rgba(255,255,255,0.85); border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; border-left: 5px solid #1E88E5; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .success-box { background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .warning-box { background-color: #FFF3E0; border-left: 5px solid #FF9800; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    div.stButton > button { background: linear-gradient(135deg, #2196F3, #9C27B0) !important; color: white !important; border: none !important; padding: 0.75rem 2rem !important; font-size: 1.2rem !important; border-radius: 50px !important; transition: all 0.3s ease !important; width: 100% !important; margin-top: 1rem !important; }
    div.stButton > button:hover { transform: scale(1.02) !important; box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4) !important; }
</style>
""", unsafe_allow_html=True)

# ---------- 首页 ----------
@safe_run
def front_page():
    set_bg_image_local("FrontPage.jpg")
    st.markdown("""
    <style>
    .right-panel { background-color: rgba(0, 0, 0, 0.70); padding: 3rem 2rem; border-radius: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5); animation: fadeIn 1s ease-in-out; color: white; }
    .right-panel h1 { text-shadow: 2px 2px 4px rgba(0,0,0,0.5); font-size: 3rem; margin-bottom: 1rem; }
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
            video_html = f"""<video width="100%" autoplay loop muted playsinline><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video>"""
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.markdown("<div style='background: rgba(255,255,255,0.2); border-radius:10px; padding:2rem; text-align:center;'><span style='font-size:3rem;'>📹</span><p style='color:white;'>Video not found.</p></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="right-panel"><h1>Welcome to<br>No-Code ML Platform</h1><p>Accessible for Machine Learning without code.</p></div>""", unsafe_allow_html=True)
        if st.button("Get Started", key="get_started", use_container_width=True):
            go_to("login")
            st.rerun()

# ---------- 登录页面 ----------
@safe_run
def login_page():
    set_bg_image_local("login.jpg")
    st.markdown("""
    <style>
    .stApp { display: flex; align-items: center; justify-content: center; }
    .stTextInput input { color: black !important; background-color: rgba(255,255,255,0.1) !important; border: 1px solid rgba(255,255,255,0.3) !important; border-radius: 5px; }
    .stTextInput label { color: black !important; }
    .back-button-container { text-align: center; margin-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='color: white; text-align: center; margin-bottom: 1.5rem;'>Login / Register</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
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
                if st.form_submit_button("Register"):
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

# ---------- 初始化会话状态 ----------
if "data" not in st.session_state:
    st.session_state.data = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "model" not in st.session_state:
    st.session_state.model = None
if "experiment" not in st.session_state:
    st.session_state.experiment = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "test_data" not in st.session_state:
    st.session_state.test_data = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "app_page" not in st.session_state:
    st.session_state.app_page = "📁 Data Upload"

# ---------- 模型名称映射 ----------
CLASSIFICATION_MODEL_MAP = {
    "Logistic Regression": "lr",
    "Random Forest": "rf",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Decision Tree": "dt",
    "Ridge": "ridge",
    "KNN": "knn",
    "SVM": "svm",
    "Naive Bayes": "nb"
}

REGRESSION_MODEL_MAP = {
    "Linear Regression": "lr",
    "Random Forest": "rf",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Decision Tree": "dt",
    "Ridge": "ridge",
    "Lasso": "lasso",
    "KNN": "knn",
    "SVM": "svm"
}

# ---------- 数据上传页面 ----------
@safe_run
def upload_page():
    st.markdown('<h2 class="sub-header">📁 Upload Your Dataset</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success(f"✔️ Loaded {len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head())
            with st.expander("📊 Basic Statistics"):
                st.write(df.describe(include='all'))
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.data is not None:
        st.markdown("### 📌 Define Target Column")
        target_col = st.selectbox("Select target column:", st.session_state.data.columns.tolist())
        problem_type = st.selectbox("Problem type:", ["Classification", "Regression"])
        if st.button("Set Target & Continue", type="primary"):
            st.session_state.target_column = target_col
            st.session_state.problem_type = problem_type
            st.session_state.app_page = "🔍 Exploratory Analysis"
            st.rerun()

# ---------- EDA 页面 ----------
@safe_run
def eda_page():
    st.markdown('<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    df = st.session_state.data
    st.dataframe(df.head())
    # 简化 EDA，只展示基本信息
    st.write("**Shape:**", df.shape)
    st.write("**Missing values:**", df.isnull().sum())

# ---------- 训练页面 ----------
@safe_run
def training_page():
    st.markdown('<h2 class="sub-header">📐 Automated Model Training with PyCaret</h2>', unsafe_allow_html=True)
    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("Please upload data and set target column first.")
        return
    if not pycaret_available:
        st.error("PyCaret not installed.")
        return

    df = st.session_state.data
    target_col = st.session_state.target_column
    problem_type = st.session_state.problem_type

    # 获取列类型
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 预处理选项（简化版本，避免过多参数导致错误）
    st.markdown("#### 🧹 Data Preprocessing")
    normalize = st.checkbox("Normalize numerical features", value=False)
    remove_outliers = st.checkbox("Remove outliers", value=False)
    feature_selection = st.checkbox("Use feature selection", value=False)
    train_size = st.slider("Training data fraction", 0.6, 0.9, 0.8)

    st.markdown("#### 🎯 Model Selection")
    if problem_type == "Classification":
        model_display_names = list(CLASSIFICATION_MODEL_MAP.keys())
    else:
        model_display_names = list(REGRESSION_MODEL_MAP.keys())
    selected_displays = st.multiselect("Include models", model_display_names, default=[])
    include_models = None if not selected_displays else [CLASSIFICATION_MODEL_MAP.get(name, name) if problem_type=="Classification" else REGRESSION_MODEL_MAP.get(name, name) for name in selected_displays]

    folds = st.slider("CV folds", 2, 10, 5)
    if problem_type == "Classification":
        metric_options = ["Accuracy", "AUC", "F1"]
    else:
        metric_options = ["R2", "MAE", "RMSE"]
    metric = st.selectbox("Optimization metric", metric_options)

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training..."):
            try:
                # 根据问题类型选择 setup
                if problem_type == "Classification":
                    setup_func = clf_setup
                    compare_func = clf_compare
                    pull_func = clf_pull
                else:
                    setup_func = reg_setup
                    compare_func = reg_compare
                    pull_func = reg_pull

                # 构建参数
                setup_params = {
                    "data": df,
                    "target": target_col,
                    "train_size": train_size,
                    "normalize": normalize,
                    "remove_outliers": remove_outliers,
                    "feature_selection": feature_selection,
                    "fold_strategy": 'kfold',
                    "n_jobs": 1,
                    "session_id": 42,
                    "verbose": False,
                    # 移除 silent 参数，避免版本问题
                }

                # 执行 setup
                exp = setup_func(**setup_params)
                st.session_state.experiment = exp

                # 比较模型
                best_model = compare_func(
                    include=include_models,
                    fold=folds,
                    sort=metric,
                    n_select=1,
                    verbose=False
                )
                st.session_state.model = best_model
                st.session_state.training_complete = True

                # 获取测试数据
                if get_config_available:
                    try:
                        X_test = get_config('X_test')
                        y_test = get_config('y_test')
                        st.session_state.test_data = {'X_test': X_test, 'y_test': y_test}
                    except:
                        st.session_state.test_data = None

                st.success("Training complete!")
                results = pull_func()
                st.dataframe(results)

            except Exception as e:
                st.error(f"Training error: {str(e)}")
                st.code(traceback.format_exc())

# ---------- 评估页面 ----------
@safe_run
def evaluation_page():
    st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)
    if not st.session_state.training_complete:
        st.warning("Train a model first.")
        return
    st.write("Model:", st.session_state.model)

# ---------- 预测页面 ----------
@safe_run
def prediction_page():
    st.markdown('<h2 class="sub-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    if not st.session_state.training_complete:
        st.warning("Train a model first.")
        return
    st.write("Upload a file for predictions.")
    # 简化

# ---------- 导出页面 ----------
@safe_run
def export_page():
    st.markdown('<h2 class="sub-header">💾 Export</h2>', unsafe_allow_html=True)
    if st.session_state.training_complete:
        st.write("Model ready for export.")

# ---------- 仪表盘 ----------
@safe_run
def dashboard_page():
    set_bg_image_local("purple.jpg")
    st.markdown(f"<h1 style='color: white;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
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

        if st.button("👋🏻 Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            go_to("front")
            st.rerun()

    # 根据选择显示页面
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

# ---------- 主入口 ----------
try:
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
except Exception as e:
    st.error("An unexpected error occurred:")
    st.code(traceback.format_exc())