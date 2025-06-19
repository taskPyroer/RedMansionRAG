# 红楼梦RAG问答系统

一个基于DeepSeek API的红楼梦智能问答系统，通过RAG（检索增强生成）技术实现精准的文本问答。

ps: 本项目仅供学习参考，代码不够优雅的地方可自行优化，代码比较简洁，易扩展，欢迎star

## 🌟 特性

- 📚 **智能文档检索**: 基于TF-IDF和余弦相似度的高效文档检索
- 🤖 **AI问答**: 集成DeepSeek API，提供准确、优雅的回答
- 🔍 **中文优化**: 使用jieba分词，专门优化中文文本处理
- 💾 **智能缓存**: 自动缓存向量索引，提升响应速度
- 🎯 **精准匹配**: 多层次文本分块，确保检索精度
- 🎨 **优雅界面**: 清晰的命令行交互界面
- 🌐 **Web界面**: 基于Streamlit的现代化交互界面
- 💬 **聊天体验**: 类ChatGPT的对话式问答体验
- 📱 **响应式设计**: 支持桌面和移动设备访问

## 📋 系统要求

- 建议Python 3.9+（本人使用3.11）
- DeepSeek API密钥
- 约50MB磁盘空间（用于依赖和缓存）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

#### 方法一：使用环境变量（推荐）

```bash
# Windows
set DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Linux/Mac
export DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

#### 方法二：使用配置文件

1. 复制配置文件模板：
```bash
copy .env.example .env
```

2. 编辑 `.env` 文件，填入您的API密钥：
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 3. 获取DeepSeek API密钥

1. 访问 [DeepSeek平台](https://platform.deepseek.com/)
2. 注册并登录账户
3. 在API管理页面创建新的API密钥
4. 复制密钥并按上述方法配置

### 4. 运行系统

#### 方式一：Streamlit Web界面（推荐）
```bash
streamlit run streamlit_app.py
```

#### 方式二：命令行界面
```bash
# 完整交互模式
python rag_system.py
```

## 使用说明

### Streamlit Web界面（推荐）

1. **启动应用**:
   ```bash
   streamlit run streamlit_app.py
   ```
   
2. **功能特性**:
   - 🎨 **现代化界面**: 美观的Web界面，支持响应式设计
   - 💬 **聊天体验**: 类似ChatGPT的对话式交互
   - ⚙️ **实时配置**: 侧边栏实时配置API密钥和系统参数
   - 📊 **系统监控**: 实时显示系统状态和文档加载情况
   - 💡 **示例问题**: 预设常见问题，一键提问
   - 📖 **文档引用**: 显示答案来源的文档片段和相似度
   - 🗑️ **历史管理**: 支持清空聊天历史

3. **使用步骤**:
   - 在侧边栏输入DeepSeek API密钥
   - 点击"初始化系统"按钮
   - 等待文档加载完成
   - 在聊天框中输入问题或点击示例问题
   - 查看AI回答和相关文档片段

### 命令行界面


#### 交互模式

运行 `rag_system.py` 进入完全交互模式：

```
=== 红楼梦智能问答系统 ===
输入您的问题，输入 'quit' 或 'exit' 退出
示例问题：
- 甄士隐是谁？
- 贾雨村的故事是什么？
- 通灵宝玉是什么？
- 英莲发生了什么事？

请输入您的问题: 
```

### 示例问答

**问题**: 甄士隐是谁？

**答案**: 甄士隐是《红楼梦》开篇的重要人物，姓甄，名费，字士隐，住在姑苏阊门外十里街仁清巷葫芦庙旁。他是一位乡宦，嫡妻封氏，情性贤淑，深明礼义。虽然家中不甚富贵，但在本地也算是望族。甄士隐禀性恬淡，不以功名为念，每日只以观花修竹、酌酒吟诗为乐，可谓神仙一流人品...

## 🏗️ 系统架构

```
红楼梦RAG系统
├── 文档加载模块
│   ├── 文档读取
│   └── 文本预处理
├── 向量化模块
│   ├── 中文分词 (jieba)
│   ├── TF-IDF向量化
│   └── 向量索引构建
├── 检索模块
│   ├── 查询向量化
│   ├── 相似度计算
│   └── 相关文档排序
└── 生成模块
    ├── 上下文构建
    ├── DeepSeek API调用
    └── 答案生成
```

## 📁 项目结构

```
RagDemos/
├── docs/                    # 文档目录
│   └── 1.txt               # 红楼梦文本
├── cache/                   # 缓存目录
│   ├── doc_chunks.pkl      # 文档块缓存
│   └── doc_vectors.pkl     # 向量索引缓存
├── rag_system.py           # 核心RAG系统
├── streamlit_app.py        # Streamlit Web应用
├── requirements.txt        # 依赖列表
├── .env.example           # 环境变量模板
├── .env                   # 环境变量文件（需要创建）
└── README.md              # 项目说明
```

## ⚙️ 配置选项

可以通过修改 `rag_system.py` 中的参数来调整系统行为：

```python
# 文档分块参数
chunk_size = 300      # 文档块大小
overlap = 50          # 重叠字符数

# 检索参数
top_k = 3             # 返回最相关的文档块数量
min_similarity = 0.01 # 最小相似度阈值

# TF-IDF参数
max_features = 5000   # 最大特征数
ngram_range = (1, 2)  # N-gram范围
```

## 🔧 高级功能

### 添加更多文档

1. 将新的文本文件放入 `docs/` 目录
2. 删除 `cache/` 目录中的缓存文件
3. 重新运行系统，会自动重建索引

### 自定义分词

可以在 `rag_system.py` 中自定义jieba分词：

```python
# 添加自定义词典
jieba.load_userdict('custom_dict.txt')

# 添加关键词
jieba.add_word('贾宝玉')
jieba.add_word('林黛玉')
```

### API参数调整

可以修改DeepSeek API调用参数：

```python
response = self.client.chat.completions.create(
    model="deepseek-chat",
    temperature=0.7,      # 创造性程度
    max_tokens=1000,      # 最大回复长度
    top_p=0.9,           # 核采样参数
    frequency_penalty=0.1 # 重复惩罚
)
```

## 🐛 故障排除

### 常见问题

1. **API密钥错误**
   ```
   错误: 401 Unauthorized
   解决: 检查API密钥是否正确设置
   ```

2. **依赖安装失败**
   ```bash
   # 升级pip
   python -m pip install --upgrade pip
   
   # 使用国内镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

3. **中文分词问题**
   ```bash
   # 重新安装jieba
   pip uninstall jieba
   pip install jieba
   ```

4. **缓存问题**
   ```bash
   # 清除缓存重建索引
   rmdir /s cache  # Windows
   rm -rf cache    # Linux/Mac
   ```

### 性能优化

1. **增加缓存**: 系统会自动缓存向量索引，首次运行较慢，后续运行会很快
2. **调整块大小**: 较小的块提供更精确的检索，较大的块提供更多上下文
3. **优化检索数量**: 增加 `top_k` 可以获得更全面的上下文，但会增加API调用成本

## 📊 系统指标

- **文档处理速度**: ~1000字符/秒
- **检索响应时间**: <100ms
- **API调用时间**: 1-3秒（取决于网络）

## 🙏 致谢

- [DeepSeek](https://www.deepseek.com/) - 提供强大的AI模型
- [jieba](https://github.com/fxsjy/jieba) - 优秀的中文分词工具
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具包
- 《红楼梦》- 中华文学瑰宝

---

**没人比我更懂红楼梦！** 🏮