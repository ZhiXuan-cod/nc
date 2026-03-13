import streamlit as st
import json
import os
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
        # 如果图片缺失，设置一个渐变色背景作为后备
        fallback_bg = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """
        st.markdown(fallback_bg, unsafe_allow_html=True)

# ---------- 用户数据存储 ----------
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

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
    div.stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
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
        background-color: #1976D2;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4);
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
            <p>Build machine learning models without code.</p>
        </div>
        """
        st.markdown(right_html, unsafe_allow_html=True)
        if st.button("Get Started", key="get_started", use_container_width=True):
            go_to("login")
            st.rerun()

# ---------- 登录/注册页面 ----------
def login_page():
    set_bg_image_local("FrontPage.jpg")
    
    # Custom CSS: style tabs, inputs, and back button
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: white;
        border-bottom-color: #2196F3;
    }
    /* Style text inputs and labels */
    .stTextInput input {
        color: white !important;
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 5px;
    }
    .stTextInput label {
        color: white !important;
    }
    /* Style the back button container */
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

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-panel">', unsafe_allow_html=True)
        st.markdown("<style>.login-panel { background-color: rgba(0, 0, 0, 0.80); padding: 2.5rem 2rem 2rem 2rem; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }</style>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='color: white; text-align: center; margin-bottom: 1.5rem;'>Login / Register</h2>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")  # removed type and style
                password = st.text_input("Password", type="password")  # keep type="password", remove style
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