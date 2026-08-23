import os

# ✅ 本地使用镜像
if "STREAMLIT_CLOUD" in os.environ or "STREAMLIT_SHARING" in os.environ:
    pass
else:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
import pandas as pd

# ==================== 加载环境变量 ====================
load_dotenv()

# ==================== DeepSeek 配置 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ==================== 敏感词检测 ====================
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
    return datetime.now().strftime("%H:%M")


def get_current_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def get_first_user_message(messages):
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    return None


def detect_sensitive_content(text):
    text_lower = text.lower()
    for keyword in SENSITIVE_KEYWORDS_LOWER:
        if keyword in text_lower:
            return True
    return False


# ==================== CSV 导出工具 ====================
def export_current_session_to_csv():
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
    if not data:
        return None, None
    df = pd.DataFrame(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename_prefix}_{timestamp}.csv"
    csv_data = df.to_csv(index=False, encoding="utf-8-sig")
    return csv_data, full_filename


# ==================== 创建 LLM ====================
def create_llm():
    """创建 DeepSeek 聊天模型"""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=0.7,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
    )


def get_response(user_input):
    """获取 DeepSeek 回复"""
    llm = create_llm()
    
    # 系统提示词
    system_prompt = """你是一名专业的心理咨询师，严格遵守中国心理咨询伦理规范。你的核心目标是建立信任关系，引导来访者探索自我、缓解困扰并促进心理成长。

# 咨询流程
1. 建立信任关系
2. 了解来访者问题
3. 分析诊断
4. 帮助指导
5. 结束关系

# 规则
- 每次对话内容长度不要超过100字，除非是进行心理问题总结或治疗方案生成
- 不要有任何动作描述，现在的场景是面对面对话，只需要输出对话内容
- 语气温和、共情、专业

# 伦理边界
- 当涉及自伤/伤人风险时，明确告知需要联系紧急联系人
- 药物治疗需由精神科医生评估，不能替代"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # 如果有历史消息，加入上下文
    if "current_messages" in st.session_state and len(st.session_state.current_messages) > 1:
        # 取最近6条历史消息作为上下文
        history = st.session_state.current_messages[-6:]
        context_messages = []
        for msg in history:
            if msg["role"] == "user":
                context_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                context_messages.append({"role": "assistant", "content": msg["content"]})
        # 重新构建消息列表
        messages = [{"role": "system", "content": system_prompt}] + context_messages + [{"role": "user", "content": user_input}]
    
    response = llm.invoke(messages)
    return response.content


# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="心理健康助手", page_icon="🧠", layout="wide")

# ==================== 初始化 Session State ====================
if "chat_sessions" not in st.session_state:
    initial_session_id = 1
    initial_messages = [
        {"role": "assistant", "content": "你好！我是心理健康助手。有什么我可以帮助你的吗？", "time": get_current_time()}
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

# ==================== 布局 ====================
col_left, col_right = st.columns([1, 3])

# ==================== 左侧边栏 ====================
with col_left:
    st.markdown("## 🧠 心灵通科技")
    st.caption("Mental Health Chatbot")
    st.divider()

    st.markdown("## 🧠 关于")
    with st.expander("📖 描述", expanded=True):
        st.markdown("基于 DeepSeek 大模型的心理健康助手，提供 24/7 心理健康支持。")
    with st.expander("🎯 目标", expanded=True):
        st.markdown("- 提供 24/7 心理健康支持\n- 危机干预\n- 连接专业资源")
    with st.expander("⚠️ 重要提示", expanded=True):
        st.markdown("仅供参考，不能替代专业医疗建议")
    st.divider()

    st.markdown("## 📋 历史对话")
    for session_id, session in sorted(st.session_state.chat_sessions.items(), key=lambda x: x[1]["id"]):
        if session_id == st.session_state.current_session_id:
            st.markdown(f"✅ **{session['title']}**")
            st.caption(f"📅 {session['date']}  |  💬 {session['message_count']} 条消息")
        else:
            if st.button(f"💬 {session['title']}", key=f"session_{session_id}", use_container_width=True):
                st.session_state.current_session_id = session_id
                st.session_state.current_messages = st.session_state.chat_sessions[session_id]["messages"].copy()
                st.rerun()

    if st.button("➕ 新建对话", use_container_width=True):
        new_id = max(st.session_state.chat_sessions.keys()) + 1 if st.session_state.chat_sessions else 1
        new_messages = [{"role": "assistant", "content": "你好！我是心理健康助手。有什么我可以帮助你的吗？", "time": get_current_time()}]
        st.session_state.chat_sessions[new_id] = {
            "id": new_id, "title": f"Chat {new_id}: 新对话", "date": get_current_date(),
            "message_count": 1, "messages": new_messages
        }
        st.session_state.current_session_id = new_id
        st.session_state.current_messages = new_messages
        st.rerun()
    st.divider()

    with st.expander("🆘 紧急求助", expanded=False):
        st.markdown(HELPLINE_NUMBERS)

    if st.button("🔄 重置当前对话", use_container_width=True):
        session_id = st.session_state.current_session_id
        new_messages = [{"role": "assistant", "content": "你好！我是心理健康助手。有什么我可以帮助你的吗？", "time": get_current_time()}]
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
                st.download_button(label="⬇️ 点击下载 CSV", data=csv_data, file_name=filename, mime="text/csv", use_container_width=True)
            else:
                st.warning("当前没有聊天记录可导出。")
        if st.button("📊 导出所有会话", use_container_width=True):
            csv_data, filename = export_all_sessions_to_csv()
            if csv_data:
                st.download_button(label="⬇️ 点击下载 CSV", data=csv_data, file_name=filename, mime="text/csv", use_container_width=True)
            else:
                st.warning("没有聊天记录可导出。")

    st.caption("© 2026 心灵通科技")
    st.caption("All Rights Reserved")

# ==================== 右侧主界面 ====================
with col_right:
    current_title = st.session_state.chat_sessions[st.session_state.current_session_id]["title"]
    st.title(f"🧠 {current_title}")
    st.caption("基于 DeepSeek 大模型，为您提供心理健康支持")

    chat_container = st.container()
    for message in st.session_state.current_messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "time" in message:
                    st.caption(f"🕐 {message['time']}")

    user_input = st.chat_input("请输入您的问题...")
    if user_input:
        current_time = get_current_time()
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
                st.caption(f"🕐 {current_time}")
            with st.chat_message("assistant"):
                if detect_sensitive_content(user_input):
                    st.warning("⚠️ 我注意到您可能正在经历困难。请记住，您并不孤单。")
                    st.markdown(HELPLINE_NUMBERS)
                    response = "我注意到您可能正在经历困难。请记住，您并不孤单，求助是一种勇气。我已为您提供了紧急求助热线。"
                else:
                    with st.spinner("思考中..."):
                        try:
                            response = get_response(user_input)
                        except Exception as e:
                            response = f"抱歉，处理您的问题时出现了错误：{e}"
                st.write(response)
                st.caption(f"🕐 {current_time}")

        st.session_state.current_messages.append({"role": "user", "content": user_input, "time": current_time})
        st.session_state.current_messages.append({"role": "assistant", "content": response, "time": current_time})

        session_id = st.session_state.current_session_id
        st.session_state.chat_sessions[session_id]["messages"] = st.session_state.current_messages.copy()
        st.session_state.chat_sessions[session_id]["message_count"] = len(st.session_state.current_messages) // 2

        first_user_msg = get_first_user_message(st.session_state.current_messages)
        if first_user_msg:
            st.session_state.chat_sessions[session_id]["title"] = f"Chat {session_id}: {first_user_msg[:20]}{'...' if len(first_user_msg) > 20 else ''}"
        st.rerun()
