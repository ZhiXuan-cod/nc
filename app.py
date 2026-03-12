import streamlit as st
import json
import os
from pathlib import Path
import base64


# ---------- 页面配置 ----------
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- 背景图片（Base64嵌入）----------
def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_image_local(image_path):
    bin_str = get_base64_of_file(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------- 用户数据存储 ----------
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register_user(email, password, name):
    users = load_users()
    if email in users:
        return False, "Email already registered."
    users[email] = {"name": name, "password": password}
    save_users(users)
    return True, "Registration successful. Please log in."

def authenticate_user(email, password):
    users = load_users()
    if email in users and users[email]["password"] == password:
        return True, users[email]["name"]
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

# ---------- 首页 Front Page（修改后）----------
def front_page():
    set_bg_image_local("FrontPage.png")

    # 自定义CSS：右侧内容块样式
    st.markdown("""
    <style>
    /* 右侧容器样式（通过key定位） */
    div[key="right-panel"] {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        animation: fadeIn 1s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* 按钮样式 */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        border-radius: 50px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
    }
    /* 左侧动画居中 */
    .st-emotion-cache-ocqkz7 {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # 两列布局：左侧动画，右侧内容
    col1, col2 = st.columns([1.2, 1.8])  # 右侧稍宽

    with col1:
        # 尝试加载Lottie动画
        try:
            from streamlit_lottie import st_lottie
            import requests
            # 机器学习主题动画（来自LottieFiles，公开资源）
            url = "https://assets2.lottiefiles.com/packages/lf20_p1qiuawe.json"  # AI动画示例
            r = requests.get(url)
            if r.status_code == 200:
                lottie_json = r.json()
                st_lottie(lottie_json, speed=1, width=300, height=300, key="ml_lottie")
            else:
                # 备用：显示本地GIF
                if os.path.exists("ml_animation.gif"):
                    st.image("ml_animation.gif", width=250)
                else:
                    st.warning("动画加载失败，请检查网络或放置 ml_animation.gif")
        except Exception as e:
            st.warning("请安装 streamlit-lottie 或提供本地动画文件")
            if os.path.exists("ml_animation.gif"):
                st.image("ml_animation.gif", width=250)

    with col2:
        # 右侧内容容器（通过key定位以便CSS应用）
        with st.container(key="right-panel"):
            st.markdown("<h1 style='color: white; font-size: 3rem; margin-bottom: 1rem;'>Welcome to<br>No-Code ML Platform</h1>", unsafe_allow_html=True)
            st.markdown("<p style='color: white; font-size: 1.2rem; opacity: 0.9;'>Build machine learning models without writing a single line of code.</p>", unsafe_allow_html=True)
            # 按钮
            if st.button("Get Started", key="get_started", use_container_width=True):
                go_to("login")
                st.rerun()

# ---------- 登录/注册页面 ----------
def login_page():
    set_bg_image_local("FrontPage.png")
    st.markdown("<h2 style='color: white; text-align: center;'>Login / Register</h2>", unsafe_allow_html=True)

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

    if st.button("← Back to Home"):
        go_to("front")
        st.rerun()

# ---------- 仪表盘 Dashboard ----------
def dashboard_page():
    set_bg_image_local("b1.png")
    st.markdown(f"<h1 style='color: white;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: white;'>This is your dashboard. Start building ML models.</p>", unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("📁 Upload Dataset")
        with col2:
            st.info("⚙️ AutoML (TPOT)")
        with col3:
            st.info("🖼️ Image Support")
        with col4:
            st.info("💬 NLP Chatbot")

    if st.button("Logout", type="primary"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        go_to("front")
        st.rerun()

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