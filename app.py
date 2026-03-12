import streamlit as st
import json
import os
from pathlib import Path

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- 自定义CSS设置背景图片 ----------
def set_bg_image(image_file):
    """通过CSS设置背景图片，image_file为图片文件名（需在同一目录）"""
    bg_css = f"""
    <style>
    .stApp {{
        background: url("./app/static/{image_file}") no-repeat center center fixed;
        background-size: cover;
    }}
    /* 可选：为内容添加半透明遮罩以提高可读性 */
    .block-container {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# 注意：Streamlit的静态文件默认从./static目录提供，但我们可以将图片放在根目录并用下面方式引用
# 更好的做法是将图片放在一个专门的static文件夹，但为简化，这里直接使用相对路径。
# 需要确保图片与app.py在同一目录，并且Streamlit能访问到。
# 由于Streamlit默认不提供任意文件的访问，我们可以通过读取图片并转换为base64来内嵌，或者将图片放在static子目录。
# 这里使用一种简单方法：将图片放在static子目录，然后通过st.image或st.markdown设置背景。
# 但为了直接设置body背景，我们需要通过CSS的url指向图片的实际访问路径。
# 在Streamlit Cloud或本地运行时，图片可以通过./media/等路径访问？更可靠的方法是使用st.image+st.markdown组合。

# 另一种更稳定的方式：使用st.image显示图片作为背景？不行。
# 最好将图片放在static文件夹，并在CSS中使用url('/static/image.png')。
# 但本地运行streamlit时，默认不会提供/static路由，除非配置。
# 我们可以使用file://协议？但不可移植。
# 简单方案：将图片转为base64嵌入CSS。

import base64

def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_image_local(image_path):
    """将本地图片转为base64并设置为背景"""
    bin_str = get_base64_of_file(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    /* 内容区域半透明背景 */
    .block-container {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 10px;
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
    users[email] = {"name": name, "password": password}  # 实际应用中密码应哈希
    save_users(users)
    return True, "Registration successful. Please log in."

def authenticate_user(email, password):
    users = load_users()
    if email in users and users[email]["password"] == password:
        return True, users[email]["name"]
    return False, None

# ---------- 页面导航 ----------
# 使用session_state管理当前页面
if "page" not in st.session_state:
    st.session_state.page = "front"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

def go_to(page):
    st.session_state.page = page

# ---------- 首页 Front Page ----------
def front_page():
    set_bg_image_local("FrontPage.png")
    st.markdown("<h1 style='color: white; text-align: center;'>Welcome to No-Code ML Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: white; text-align: center; font-size: 1.2em;'>Build machine learning models without writing a single line of code.</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Get Started", use_container_width=True):
            go_to("login")

# ---------- 登录/注册页面 ----------
def login_page():
    set_bg_image_local("FrontPage.png")  # 登录页也使用首页背景，或可更换
    st.markdown("<h2 style='color: white; text-align: center;'>Login / Register</h2>", unsafe_allow_html=True)

    # 使用 tabs 区分登录和注册
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

    # 返回首页链接
    if st.button("← Back to Home"):
        go_to("front")
        st.rerun()

# ---------- 仪表盘 Dashboard ----------
def dashboard_page():
    set_bg_image_local("b1.png")
    st.markdown(f"<h1 style='color: white;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: white;'>This is your dashboard. Start building ML models.</p>", unsafe_allow_html=True)

    # 模拟功能卡片（可根据PDF后续扩展）
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