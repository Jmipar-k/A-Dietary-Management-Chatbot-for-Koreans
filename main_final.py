import streamlit as st
from PIL import Image
from agent import load_agent
import base64

# 배경 이미지 설정 함수
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit 페이지 설정
st.set_page_config(page_title="Healthy Diet Advisor", page_icon="🍎", layout="wide")
set_background("./ui_background.png")  # 배경 이미지 파일 경로

# CSS 스타일링
st.markdown(
    """
    <style>
    .title {
        font-size: 2em;
        color: #FF6B6B;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .response-box {
        background-color: #f0f2f6;
        border: 1px solid #d3d3d3;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        line-height: 1.5;
        color: #333333;
        font-size: 1.1em;
    }
    .uploaded-image img {
        width: 244px;
        height: 244px;
        object-fit: cover;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 타이틀
st.markdown('<p class="title">🥗 Healthy Diet Chatbot 🥗</p>', unsafe_allow_html=True)

# 세션 상태 초기화
if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = load_agent()
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 사이드바: 대화 기록 초기화 버튼
with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Reset Conversation"):
        st.session_state.conversation = []
        st.success("Conversation has been reset successfully!")

# 이전 대화 기록 출력
# st.write("### Conversation History")
for entry in st.session_state.conversation:
    st.markdown(
        f"""
        <div class="response-box">
            <p><strong>You:</strong> {entry['user']}</p>
            <p><strong>Advisor:</strong> {entry['assistant']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # 이미지 출력 (기록에 이미지가 있을 경우)
    if entry.get("image"):
        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(entry["image"], caption="Uploaded Image", width=244)
        st.markdown('</div>', unsafe_allow_html=True)

# 입력창과 이미지 업로드
uploaded_image = st.file_uploader("📷 Upload a food image", type=["jpg", "jpeg", "png"], key="image_uploader")
user_question = st.text_input("💬 Enter your question here!", key="question_input")
submit_button = st.button("🚀 Get Advice")

# 질문 및 이미지 처리
if submit_button and user_question:
    # 에이전트 실행
    with st.spinner("Processing your question..."):
        inputs = {
            "image": Image.open(uploaded_image).convert("RGB") if uploaded_image else None,
            "question": user_question
        }
        result = st.session_state.agent_chain.run(inputs)

    # 대화 기록 저장 (이미지 포함)
    st.session_state.conversation.append({
        "user": user_question,
        "assistant": result,
        "image": uploaded_image if uploaded_image else None
    })

    # 입력창과 이미지 초기화
    st.rerun()
