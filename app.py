import os

# ✅ 智能判断：云端用官方源，本地用镜像
if "STREAMLIT_CLOUD" in os.environ or "STREAMLIT_SHARING" in os.environ:
    # Streamlit Cloud 服务器在国外，用官方源更快
    pass
else:
    # 本地开发，使用国内镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # ✅ 移到顶部
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
import pandas as pd

# ==================== 加载环境变量 ====================
load_dotenv()

# ==================== DeepSeek 配置 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

PERSIST_DIRECTORY = "./vector_store"
DATA_PATH = "./data"

# ==================== 敏感词检测（预处理版本） ====================
SENSITIVE_KEYWORDS = [
    "自杀", "自残", "自我伤害", "想死", "不想活了",
    "割腕", "跳楼", "自伤", "抑郁", "绝望",
    "self-harm", "suicide", "kill myself"
]
SENSITIVE_KEYWORDS_LOWER = [kw.lower() for kw in SENSITIVE_KEYWORDS]

HELPLINE_NUMBERS = """
### 🆘 紧急求助热线

如果您或您认识的人正在经历心理危机，请立即联系以下热线：

- **全国心理援助热线**：400-161-9995
- **希望24热线**：400-161-9995
- **北京心理危机干预中心**：010-82951332

*请记住：您并不孤单，求助是一种勇气。*
"""


# ==================== 工具函数 ====================
def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%H:%M")


def get_current_date():
    """获取当前日期时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def get_first_user_message(messages):
    """获取当前会话的第一条用户消息"""
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    return None


def detect_sensitive_content(text):
    """检测是否包含敏感内容"""
    text_lower = text.lower()
    for keyword in SENSITIVE_KEYWORDS_LOWER:
        if keyword in text_lower:
            return True
    return False


# ==================== CSV 导出工具 ====================
def export_current_session_to_csv():
    """导出当前会话为 CSV"""
    if "current_messages" not in st.session_state or not st.session_state.current_messages:
        return None, None

    data = []
    for i, msg in enumerate(st.session_state.current_messages, 1):
        data.append({
            "序号": i,
            "角色": "用户" if msg["role"] == "user" else "助手",
            "内容": msg["content"],
            "时间": msg.get("time", "")
        })
    return _export_to_csv(data, f"会话_{st.session_state.current_session_id}")


def export_all_sessions_to_csv():
    """导出所有会话为 CSV"""
    if "chat_sessions" not in st.session_state or not st.session_state.chat_sessions:
        return None, None

    data = []
    for session_id, session in st.session_state.chat_sessions.items():
        for msg in session["messages"]:
            data.append({
                "会话ID": session_id,
                "会话标题": session["title"],
                "日期": session["date"],
                "角色": "用户" if msg["role"] == "user" else "助手",
                "内容": msg["content"],
                "时间": msg.get("time", "")
            })
    return _export_to_csv(data, "全部聊天记录")


def _export_to_csv(data, filename_prefix):
    """通用 CSV 导出函数"""
    if not data:
        return None, None
    df = pd.DataFrame(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename_prefix}_{timestamp}.csv"
    csv_data = df.to_csv(index=False, encoding="utf-8-sig")
    return csv_data, full_filename


# ==================== RAG 核心功能 ====================
@st.cache_resource
def load_and_process_documents():
    """加载并分割文档"""
    documents = []

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return []

    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
        except Exception as e:
            st.error(f"❌ 加载 {file} 失败: {e}")

    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


@st.cache_resource
def create_vector_store(chunks):
    """创建向量数据库"""
    if not chunks:
        return None

    # ✅ 使用 FastEmbed（轻量级，无下载依赖）
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    if os.path.exists(PERSIST_DIRECTORY):
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        if vector_store._collection.count() > 0:
            return vector_store

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vector_store


def create_rag_chain(vector_store):
    """创建 RAG 问答链"""
    if not vector_store:
        return None

    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=0.3,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
    )

    prompt_template = """
    你是一个专业的心理健康助理。请基于以下上下文信息回答用户的问题。

    如果上下文中包含求助热线信息，请务必提供给用户。

    上下文：
    {context}

    用户问题：{question}

    回答：
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain


# ==================== Streamlit 界面 ====================
st.set_page_config(
    page_title="心理健康助手",
    page_icon="🧠",
    layout="wide"
)

# ==================== 初始化 Session State ====================
if "chat_sessions" not in st.session_state:
    initial_session_id = 1
    initial_messages = [
        {
            "role": "assistant",
            "content": "你好！我是心理健康助手。有什么我可以帮助你的吗？",
            "time": get_current_time()
        }
    ]

    st.session_state.chat_sessions = {
        initial_session_id: {
            "id": initial_session_id,
            "title": "Chat 1: 初次咨询",
            "date": get_current_date(),
            "message_count": 1,
            "messages": initial_messages
        }
    }

    st.session_state.current_session_id = initial_session_id
    st.session_state.current_messages = initial_messages.copy()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = 1

if "current_messages" not in st.session_state:
    current_id = st.session_state.current_session_id
    st.session_state.current_messages = st.session_state.chat_sessions[current_id]["messages"].copy()

if "qa_chain" not in st.session_state:
    with st.spinner("正在加载心理健康数据..."):
        chunks = load_and_process_documents()
        if chunks:
            vector_store = create_vector_store(chunks)
            st.session_state.qa_chain = create_rag_chain(vector_store)
        else:
            st.session_state.qa_chain = None

# ==================== 布局 ====================
col_left, col_right = st.columns([1, 3])

# ==================== 左侧边栏 ====================
with col_left:
    # 顶部团队名称
    st.markdown("## 🧠 团队名称")
    st.caption("Mental Health Chatbot")
    st.divider()

    # 关于部分
    st.markdown("## 🧠 关于")
    with st.expander("📖 描述", expanded=True):
        st.markdown("""
        一个基于 RAG 技术的心理健康助手，为用户提供可靠的心理健康信息和支持。
        """)
    with st.expander("🎯 目标", expanded=True):
        st.markdown("""
        - 提供 24/7 心理健康支持
        - 危机干预
        - 连接专业资源
        """)
    with st.expander("📚 数据来源", expanded=True):
        st.markdown("""
        - 世界卫生组织 (WHO)
        - 心理健康指南
        """)
    with st.expander("⚠️ 重要提示", expanded=True):
        st.markdown("仅供参考，不能替代专业医疗建议")

    st.divider()

    # 历史对话
    st.markdown("## 📋 历史对话")
    for session_id, session in sorted(st.session_state.chat_sessions.items(), key=lambda x: x[1]["id"]):
        if session_id == st.session_state.current_session_id:
            st.markdown(f"✅ **{session['title']}**")
            st.caption(f"📅 {session['date']}  |  💬 {session['message_count']} 条消息")
        else:
            if st.button(
                    f"💬 {session['title']}",
                    key=f"session_{session_id}",
                    use_container_width=True
            ):
                st.session_state.current_session_id = session_id
                st.session_state.current_messages = st.session_state.chat_sessions[session_id]["messages"].copy()
                st.rerun()

    # 新建对话
    if st.button("➕ 新建对话", use_container_width=True):
        new_id = max(st.session_state.chat_sessions.keys()) + 1 if st.session_state.chat_sessions else 1
        new_messages = [
            {
                "role": "assistant",
                "content": "你好！我是心理健康助手。有什么我可以帮助你的吗？",
                "time": get_current_time()
            }
        ]
        st.session_state.chat_sessions[new_id] = {
            "id": new_id,
            "title": f"Chat {new_id}: 新对话",
            "date": get_current_date(),
            "message_count": 1,
            "messages": new_messages
        }
        st.session_state.current_session_id = new_id
        st.session_state.current_messages = new_messages
        st.rerun()

    st.divider()

    # 紧急求助、重置、导出
    with st.expander("🆘 紧急求助", expanded=False):
        st.markdown(HELPLINE_NUMBERS)

    if st.button("🔄 重置当前对话", use_container_width=True):
        session_id = st.session_state.current_session_id
        new_messages = [
            {
                "role": "assistant",
                "content": "你好！我是心理健康助手。有什么我可以帮助你的吗？",
                "time": get_current_time()
            }
        ]
        st.session_state.current_messages = new_messages
        st.session_state.chat_sessions[session_id]["messages"] = new_messages
        st.session_state.chat_sessions[session_id]["message_count"] = 1
        st.session_state.chat_sessions[session_id]["title"] = f"Chat {session_id}: 新对话"
        st.rerun()

    with st.expander("📤 导出聊天记录", expanded=False):
        st.markdown("导出当前会话或所有会话的聊天记录为 CSV 文件。")

        if st.button("📄 导出当前会话", use_container_width=True):
            csv_data, filename = export_current_session_to_csv()
            if csv_data:
                st.download_button(
                    label="⬇️ 点击下载 CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("当前没有聊天记录可导出。")

        if st.button("📊 导出所有会话", use_container_width=True):
            csv_data, filename = export_all_sessions_to_csv()
            if csv_data:
                st.download_button(
                    label="⬇️ 点击下载 CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("没有聊天记录可导出。")

    # 底部页脚
    st.caption("© 2026 心灵通科技")
    st.caption("All Rights Reserved")

# ==================== 右侧主界面 ====================
with col_right:
    current_title = st.session_state.chat_sessions[st.session_state.current_session_id]["title"]
    st.title(f"🧠 {current_title}")
    st.caption("基于 RAG 技术，为您提供可靠的心理健康信息")

    # 聊天界面
    chat_container = st.container()

    for message in st.session_state.current_messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "time" in message:
                    st.caption(f"🕐 {message['time']}")

    # 用户输入
    user_input = st.chat_input("请输入您的问题...")

    if user_input:
        current_time = get_current_time()

        with chat_container:
            # 用户消息
            with st.chat_message("user"):
                st.write(user_input)
                st.caption(f"🕐 {current_time}")

            # 助手消息
            with st.chat_message("assistant"):
                if detect_sensitive_content(user_input):
                    st.warning("⚠️ 我注意到您可能正在经历困难。请记住，您并不孤单。")
                    st.markdown(HELPLINE_NUMBERS)
                    response = "我注意到您可能正在经历困难。请记住，您并不孤单，求助是一种勇气。我已为您提供了紧急求助热线。"
                else:
                    with st.spinner("思考中..."):
                        if st.session_state.qa_chain:
                            try:
                                result = st.session_state.qa_chain.invoke({"query": user_input})
                                response = result["result"]
                                if "热线" in response or "求助" in response:
                                    st.markdown(HELPLINE_NUMBERS)
                            except Exception as e:
                                response = f"抱歉，处理您的问题时出现了错误：{e}"
                        else:
                            response = "抱歉，我还没有加载心理健康数据。请确保 `data` 文件夹中有 PDF 或 TXT 文件。"

                st.write(response)
                st.caption(f"🕐 {current_time}")

        # 保存消息
        st.session_state.current_messages.append({
            "role": "user",
            "content": user_input,
            "time": current_time
        })
        st.session_state.current_messages.append({
            "role": "assistant",
            "content": response,
            "time": current_time
        })

        # 同步到会话存储
        session_id = st.session_state.current_session_id
        st.session_state.chat_sessions[session_id]["messages"] = st.session_state.current_messages.copy()
        st.session_state.chat_sessions[session_id]["message_count"] = len(st.session_state.current_messages) // 2

        # 更新标题
        first_user_msg = get_first_user_message(st.session_state.current_messages)
        if first_user_msg:
            st.session_state.chat_sessions[session_id][
                "title"] = f"Chat {session_id}: {first_user_msg[:20]}{'...' if len(first_user_msg) > 20 else ''}"

        st.rerun()
