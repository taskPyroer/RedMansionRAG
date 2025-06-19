# -*- coding: utf-8 -*-
"""
红楼梦RAG问答系统 - Streamlit交互界面
基于DeepSeek API实现智能问答功能
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from rag_system import RedMansionRAG

# 页面配置
st.set_page_config(
    page_title="红楼梦RAG问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载环境变量
load_dotenv()

# 自定义CSS样式
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&display=swap');

.main-header {
    font-size: 3rem;
    background: linear-gradient(135deg, #8B4513, #CD853F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
    font-family: 'Noto Serif SC', serif;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.subtitle {
    text-align: center;
    color: #666;
    font-style: italic;
    margin-bottom: 2rem;
    font-family: 'Noto Serif SC', serif;
}

.chat-message {
    padding: 1.2rem;
    border-radius: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    animation: fadeIn 0.5s ease-in;
}

.chat-message:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left: 4px solid #2196f3;
    margin-left: 2rem;
}

.assistant-message {
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border-left: 4px solid #8B4513;
    margin-right: 2rem;
}

.source-info {
    background: linear-gradient(135deg, #f3e5f5, #e1bee7);
    padding: 0.8rem;
    border-radius: 0.8rem;
    margin-top: 0.8rem;
    font-size: 0.9rem;
    border: 1px solid #ce93d8;
    transition: all 0.3s ease;
}

.source-info:hover {
    background: linear-gradient(135deg, #e8eaf6, #c5cae9);
    border-color: #9c27b0;
}

.status-success {
    color: #2e7d32;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.status-error {
    color: #d32f2f;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.status-warning {
    color: #f57c00;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.example-button {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    border: 1px solid #ffb74d;
    border-radius: 0.5rem;
    padding: 0.5rem;
    margin: 0.2rem 0;
    transition: all 0.3s ease;
    cursor: pointer;
}

.example-button:hover {
    background: linear-gradient(135deg, #ffe0b2, #ffcc80);
    transform: translateX(5px);
}

.sidebar-section {
    background: linear-gradient(135deg, #fafafa, #f5f5f5);
    padding: 1rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    border: 1px solid #e0e0e0;
}

.typing-indicator {
    display: inline-block;
    animation: typing 1.5s infinite;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes typing {
    0%, 60%, 100% { opacity: 1; }
    30% { opacity: 0.5; }
}

.metric-card {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    padding: 1rem;
    border-radius: 0.8rem;
    text-align: center;
    margin: 0.5rem 0;
    border: 1px solid #81c784;
}

.progress-bar {
    background: linear-gradient(90deg, #4caf50, #8bc34a);
    height: 4px;
    border-radius: 2px;
    animation: progress 2s ease-in-out;
}

@keyframes progress {
    from { width: 0%; }
    to { width: 100%; }
}
</style>
""", unsafe_allow_html=True)

def check_cache_exists():
    """检查缓存文件是否存在"""
    cache_dir = Path("cache")
    vectors_cache = cache_dir / "doc_vectors.pkl"
    chunks_cache = cache_dir / "doc_chunks.pkl"
    return vectors_cache.exists() and chunks_cache.exists()

def save_api_key_to_env(api_key):
    """保存API密钥到.env文件"""
    env_file = Path(".env")
    
    # 读取现有的.env文件内容
    env_lines = []
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            env_lines = f.readlines()
    
    # 查找并更新DEEPSEEK_API_KEY行
    api_key_updated = False
    for i, line in enumerate(env_lines):
        if line.strip().startswith('DEEPSEEK_API_KEY='):
            env_lines[i] = f'DEEPSEEK_API_KEY={api_key}\n'
            api_key_updated = True
            break
    
    # 如果没有找到，则添加新行
    if not api_key_updated:
        env_lines.append(f'DEEPSEEK_API_KEY={api_key}\n')
    
    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(env_lines)

def auto_initialize_from_cache(api_key):
    """从缓存自动初始化系统"""
    if not api_key:
        return False
    try:
        rag = RedMansionRAG(api_key=api_key)
        rag.initialize()
        st.session_state.rag_system = rag
        st.session_state.system_initialized = True
        st.session_state.auto_initialized = True
        return True
    except Exception as e:
        # 改进错误处理，确保即使异常对象没有合适的字符串表示也能提供有用的错误信息
        error_type = type(e).__name__
        error_details = str(e) if str(e) else "未知错误"
        error_msg = f'自动初始化失败: {error_type} - {error_details}'
        
        # 记录详细错误信息用于调试
        print(f"Error in auto initialization: {error_type}: {error_details}")
        
        st.error(error_msg)
        return False

def initialize_session_state():
    """初始化会话状态"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('DEEPSEEK_API_KEY', '')
    if 'temp_api_key' not in st.session_state:
        st.session_state.temp_api_key = ''
    if 'save_api_key_option' not in st.session_state:
        st.session_state.save_api_key_option = False
    if 'auto_initialized' not in st.session_state:
        st.session_state.auto_initialized = False
    if 'preset_question' not in st.session_state:
        st.session_state.preset_question = ""
    if 'selected_role' not in st.session_state:
        st.session_state.selected_role = ""
    
    # 检查缓存并自动初始化
    # 优先使用临时密钥，如果没有则使用持久化密钥
    current_api_key = st.session_state.temp_api_key or st.session_state.api_key
    if (not st.session_state.system_initialized and 
        current_api_key and 
        check_cache_exists()):
        auto_initialize_from_cache(current_api_key)

def display_chat_message(role, content, sources=None, typing=False):
    """显示聊天消息"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋‍♀️ 您:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        typing_indicator = '<span class="typing-indicator">💭</span>' if typing else '🤖'
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>{typing_indicator} 红楼梦助手:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            with st.expander(f"📖 参考文档片段 ({len(sources)}个)", expanded=False):
                for i, source in enumerate(sources, 1):
                    similarity_color = "#4caf50" if source['similarity'] > 0.5 else "#ff9800" if source['similarity'] > 0.3 else "#f44336"
                    st.markdown(f"""
                    <div class="source-info">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong>📄 片段 {i}: {source['source']}</strong>
                            <span style="background: {similarity_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 1rem; font-size: 0.8rem;">
                                相似度: {source['similarity']:.3f}
                            </span>
                        </div>
                        <div style="background: rgba(255,255,255,0.8); padding: 0.5rem; border-radius: 0.5rem; border-left: 3px solid {similarity_color};">
                            <em>📝 内容预览:</em><br>
                            {source['content_preview']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def initialize_rag_system(api_key):
    """初始化RAG系统"""
    with st.spinner('正在初始化红楼梦RAG系统...'):
        rag = RedMansionRAG(api_key=api_key)
        rag.initialize()
        st.session_state.rag_system = rag
        st.session_state.system_initialized = True
        st.success('✅ 系统初始化成功！')
        return True

def main():
    """主函数"""
    initialize_session_state()
    
    # 页面标题
    st.markdown('<h1 class="main-header">📚 红楼梦RAG问答系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">"满纸荒唐言，一把辛酸泪。都云作者痴，谁解其中味？"</p>', unsafe_allow_html=True)
    
    # 添加进度条效果
    if st.session_state.system_initialized:
        st.markdown('<div class="progress-bar"></div>', unsafe_allow_html=True)
    
    # 侧边栏配置
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ 系统配置")
        
        # API密钥配置
        # 显示逻辑：优先显示临时密钥，如果没有则显示持久化密钥
        display_key = st.session_state.get('temp_api_key', '') or st.session_state.api_key
        api_key_input = st.text_input(
            "🔑 DeepSeek API密钥",
            value=display_key,
            type="password",
            help="请输入您的DeepSeek API密钥。如已保存到配置文件，刷新页面后会自动加载。",
            placeholder="sk-xxxxxxxxxxxxxxxx"
        )
        
        # 添加保存选项
        save_api_key = st.checkbox(
            "💾 保存API密钥到配置文件",
            value=False,
            help="勾选此项将把API密钥保存到.env文件中，下次启动时自动加载"
        )
        
        # 处理API密钥变化
        api_key_changed = api_key_input != st.session_state.get('temp_api_key', '')
        save_option_changed = save_api_key != st.session_state.get('save_api_key_option', False)
        
        if api_key_changed or save_option_changed:
            # 更新临时密钥
            st.session_state.temp_api_key = api_key_input
            st.session_state.save_api_key_option = save_api_key
            
            # 如果勾选了保存且有API密钥，则保存到文件
            if save_api_key and api_key_input:
                st.session_state.api_key = api_key_input
                # 保存到.env文件
                save_api_key_to_env(api_key_input)
                st.success("✅ API密钥已保存到配置文件")
            elif not save_api_key:
                # 如果没有勾选保存，使用临时密钥但不持久化
                st.session_state.api_key = api_key_input
            
            if api_key_changed:
                st.session_state.system_initialized = False
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 系统状态
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📊 系统状态")
        if st.session_state.system_initialized:
            st.markdown('<p class="status-success">✅ 系统已就绪</p>', unsafe_allow_html=True)
            if st.session_state.rag_system:
                # 使用metric卡片显示统计信息
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>📄</h3>
                        <p>{len(st.session_state.rag_system.documents)}</p>
                        <small>文档数量</small>
                    </div>
                    ''', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>📝</h3>
                        <p>{len(st.session_state.rag_system.doc_chunks)}</p>
                        <small>文档块</small>
                    </div>
                    ''', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⚠️ 系统未初始化</p>', unsafe_allow_html=True)
        
        # 初始化按钮
        cache_exists = check_cache_exists()
        # 获取当前有效的API密钥
        current_api_key = st.session_state.temp_api_key or st.session_state.api_key
        if cache_exists and st.session_state.system_initialized:
            if st.button("🔄 重新初始化系统", use_container_width=True):
                if current_api_key:
                    st.session_state.system_initialized = False
                    st.session_state.rag_system = None
                    initialize_rag_system(current_api_key)
                else:
                    st.error("请先输入API密钥")
        else:
            button_text = "🚀 初始化系统" if not cache_exists else "🚀 加载缓存数据"
            if st.button(button_text, disabled=not current_api_key, use_container_width=True):
                if current_api_key:
                    initialize_rag_system(current_api_key)
                else:
                    st.error("请先输入API密钥")
        
        # 清空聊天历史
        if st.button("🗑️ 清空聊天历史", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 搜索参数配置
        if st.session_state.system_initialized:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.subheader("🔍 搜索参数")
            
            # 最大返回文档数量
            top_k = st.slider(
                "📄 最大返回文档数量",
                min_value=1,
                max_value=20,
                value=10,
                help="设置搜索时最多返回多少个相关文档片段"
            )
            
            # 相似度阈值
            similarity_threshold = st.slider(
                "📊 相似度阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.01,
                help="只返回相似度高于此值的文档片段，值越高结果越精确"
            )
            
            # 保存到session state
            st.session_state.search_top_k = top_k
            st.session_state.search_similarity_threshold = similarity_threshold
            
            st.markdown('</div>', unsafe_allow_html=True)

    # 主要内容区域
    # 获取当前有效的API密钥（优先使用临时密钥）
    current_api_key = st.session_state.temp_api_key or st.session_state.api_key
    
    if not current_api_key:
        st.warning("⚠️ 请在侧边栏输入DeepSeek API密钥以开始使用")
        st.info("""
        ### 如何获取API密钥：
        1. 访问 [DeepSeek平台](https://platform.deepseek.com/)
        2. 注册并登录账户
        3. 在API管理页面创建新的API密钥
        4. 将密钥粘贴到左侧输入框中
        
        💡 **提示**: 默认情况下，API密钥仅在当前会话中临时缓存，不会保存到文件。如需持久化保存，请勾选"保存API密钥到配置文件"选项。
        """)
        return
    
    if not st.session_state.system_initialized:
        cache_exists = check_cache_exists()
        if cache_exists:
            if st.session_state.auto_initialized:
                st.success("✅ 系统已从缓存自动加载完成！")
            else:
                st.info("💡 检测到缓存文件，请点击侧边栏的'加载缓存数据'按钮快速启动系统")
        else:
            st.info("💡 请点击侧边栏的'初始化系统'按钮来加载红楼梦文档")
        return
    
    # 主要对话区域
    st.header("💬 与红楼梦助手对话")
    
    # 聊天界面 - 显示聊天历史（放在最上方）
    st.subheader("📜 历史对话")
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                display_chat_message("user", message["content"])
            else:
                display_chat_message("assistant", message["content"], message.get("sources"))
    
    st.markdown("---")
    
    # 添加快捷操作按钮（放在输入框上方）
    st.subheader("⚡ 快捷问题")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎭 人物关系", use_container_width=True):
            if st.session_state.system_initialized:
                # 设置预选问题到session state
                st.session_state.preset_question = "红楼梦中主要人物之间的关系是怎样的？"
                st.session_state.selected_role = "人物关系"
                st.rerun()
    with col2:
        if st.button("📚 情节梗概", use_container_width=True):
            if st.session_state.system_initialized:
                st.session_state.preset_question = "请简述红楼梦第一回的主要情节。"
                st.session_state.selected_role = "情节梗概"
                st.rerun()
    with col3:
        if st.button("🎨 文学手法", use_container_width=True):
            if st.session_state.system_initialized:
                st.session_state.preset_question = "红楼梦运用了哪些文学表现手法？"
                st.session_state.selected_role = "文学手法"
                st.rerun()
    with col4:
        if st.button("💎 象征意义", use_container_width=True):
            if st.session_state.system_initialized:
                st.session_state.preset_question = "通灵宝玉有什么象征意义？"
                st.session_state.selected_role = "象征意义"
                st.rerun()
    
    st.markdown("---")
    
    # 显示已选择的角色（如果有）
    if st.session_state.selected_role:
        st.info(f"✅ 已选择角色：{st.session_state.selected_role}")
    
    # 用户输入区域（放在最下方）
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            # 使用预设问题作为默认值
            user_input = st.text_area(
                "💭 请输入您的问题：",
                value=st.session_state.preset_question,  # 使用预设问题
                placeholder="例如：甄士隐是谁？他在故事中起到什么作用？\n\n💡 提示：您可以询问人物、情节、诗词、象征意义等任何关于红楼梦的问题",
                height=100,
                label_visibility="visible"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # 添加间距
            submit_button = st.form_submit_button("🚀 发送", use_container_width=True, type="primary")
            clear_button = st.form_submit_button("🧹 清空", use_container_width=True)

    # 处理用户输入
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.preset_question = ""
        st.session_state.selected_role = ""
        st.rerun()
    
    if submit_button and user_input.strip():
        if st.session_state.system_initialized and st.session_state.rag_system:
            # 添加用户消息到历史
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # 清除预设问题和选择的角色
            st.session_state.preset_question = ""
            st.session_state.selected_role = ""
            
            # 显示用户消息
            display_chat_message("user", user_input)
            
            # 显示思考中的状态
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                display_chat_message("assistant", "正在思考中，请稍候...", typing=True)
            
            # 生成回答
            try:
                with st.spinner('🔍 正在搜索相关文档... 📚 正在生成回答...'):
                    # 获取搜索参数
                    top_k = getattr(st.session_state, 'search_top_k', 10)
                    similarity_threshold = getattr(st.session_state, 'search_similarity_threshold', 0.01)
                    result = st.session_state.rag_system.ask(user_input, top_k=top_k, similarity_threshold=similarity_threshold)
                
                # 清除思考状态
                thinking_placeholder.empty()
                
                # 处理来源信息
                processed_sources = []
                for source in result['sources']:
                    # 确保使用正确的字段名称
                    if 'content_preview' in source:
                        content = source['content_preview']
                    elif 'content' in source:
                        content = source['content'][:200] + '...' if len(source['content']) > 200 else source['content']
                    else:
                        content = "无内容预览"
                        
                    processed_sources.append({
                        'source': source['source'],
                        'similarity': source['similarity'],
                        'content_preview': content
                    })
                
                # 添加助手回答到历史
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": result['answer'],
                    "sources": processed_sources
                })
                
                # 显示助手回答
                display_chat_message("assistant", result['answer'], processed_sources)
                

                
            except Exception as e:
                thinking_placeholder.empty()
                # 改进错误处理，确保即使异常对象没有合适的字符串表示也能提供有用的错误信息
                error_type = type(e).__name__
                error_details = str(e) if str(e) else "未知错误"
                error_msg = f"抱歉，处理您的问题时出现错误：{error_type} - {error_details}"
                
                # 记录详细错误信息用于调试
                print(f"Error in ask function: {error_type}: {error_details}")
                
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                display_chat_message("assistant", error_msg)
                
                # 显示错误详情（仅在开发模式下）
                with st.expander("🔧 错误详情（开发调试用）", expanded=False):
                    st.code(f"错误类型: {error_type}\n错误信息: {error_details}\n")
                    if hasattr(e, '__traceback__'):
                        import traceback
                        st.code(traceback.format_exc())
            
            # 重新运行以更新界面
            st.rerun()
        else:
            st.error("❌ 系统未初始化，请先在侧边栏初始化系统")
    
    # 页面底部信息
    st.markdown("---")
    
    # 添加使用统计
    if st.session_state.system_initialized and st.session_state.rag_system:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💬 对话轮次", len([msg for msg in st.session_state.chat_history if msg['role'] == 'user']))
        with col2:
            st.metric("📄 文档数量", len(st.session_state.rag_system.documents))
        with col3:
            st.metric("📝 文档块数", len(st.session_state.rag_system.doc_chunks))
        with col4:
            st.metric("🤖 AI模型", "DeepSeek")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 功能介绍
    with st.expander("ℹ️ 系统功能介绍", expanded=False):
        st.markdown("""
        ### 🌟 主要功能
        - **🔍 智能检索**: 基于TF-IDF向量化的语义搜索
        - **🤖 AI问答**: 集成DeepSeek API的智能回答生成
        - **📚 文档管理**: 自动加载和处理红楼梦文本
        - **⚡ 缓存优化**: 智能缓存机制，提升响应速度
        - **💬 聊天体验**: 类ChatGPT的对话式交互
        - **📖 文档溯源**: 显示答案来源和相似度评分
        
        ### 💡 使用技巧
        - 可以询问人物、情节、诗词、象征意义等任何关于红楼梦的问题
        - 使用侧边栏的示例问题快速开始
        - 查看参考文档片段了解答案来源
        - 使用快捷操作按钮探索不同主题
        """)
    
    # 底部版权信息
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
    ">
        <h4 style="color: #8B4513; margin-bottom: 1rem;">📚 红楼梦RAG问答系统</h4>
        <p style="color: #666; margin-bottom: 1rem;">
            基于 <strong>DeepSeek API</strong> 构建 | 采用 <strong>RAG技术</strong> | 使用 <strong>Streamlit</strong> 开发
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span style="color: #555;">🔗 <a href="https://platform.deepseek.com/" target="_blank" style="color: #8B4513; text-decoration: none;">DeepSeek平台</a></span>
            <span style="color: #555;">📖 <a href="https://streamlit.io/" target="_blank" style="color: #8B4513; text-decoration: none;">Streamlit官网</a></span>
            <span style="color: #555;">💻 <a href="https://github.com/taskPyroer/RedMansionRAG" target="_blank" style="color: #8B4513; text-decoration: none;">GitHub源码</a></span>
            <span style="color: #555;">🔗 <a href="https://docs.taskpyro.cn/assets/mp-qr-xRjY1oQw.png" target="_blank" style="color: #8B4513; text-decoration: none;">公众号</a></span>
        </div>
        <p style="color: #888; font-size: 0.8rem; margin-top: 1rem;">
            "满纸荒唐言，一把辛酸泪。都云作者痴，谁解其中味？" - 曹雪芹
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()