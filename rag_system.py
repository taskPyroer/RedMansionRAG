# -*- coding: utf-8 -*-
"""
红楼梦RAG问答系统
基于DeepSeek API实现智能问答功能
"""

import os
import re
import pickle
from typing import List, Dict
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.analyse
from dotenv import load_dotenv

class RedMansionRAG:
    """红楼梦RAG问答系统"""
    
    def __init__(self, api_key: str, docs_dir: str = "docs"):
        """
        初始化RAG系统
        
        Args:
            api_key: DeepSeek API密钥
            docs_dir: 文档目录路径
        """
        self.api_key = api_key
        self.docs_dir = Path(docs_dir)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 文档存储
        self.documents = []
        self.doc_chunks = []
        self.vectorizer = None
        self.doc_vectors = None
        
        # 缓存文件路径
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.vectors_cache = self.cache_dir / "doc_vectors.pkl"
        self.chunks_cache = self.cache_dir / "doc_chunks.pkl"
        
        # 初始化jieba
        jieba.initialize()
        
        # 加载中文停用词库
        self.stopwords = self.load_stopwords()
        
    def load_documents(self) -> None:
        """加载文档"""
        print("正在加载红楼梦文档...")
        
        for file_path in self.docs_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.documents.append({
                            'filename': file_path.name,
                            'content': content,
                            'path': str(file_path)
                        })
                        print(f"已加载: {file_path.name}")
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
        
        print(f"共加载 {len(self.documents)} 个文档")
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """将文本分割成块"""
        # 按句号、问号、感叹号分句
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def preprocess_documents(self) -> None:
        """预处理文档，分割成块"""
        print("正在预处理文档...")
        
        # 检查缓存
        if self.chunks_cache.exists():
            print("发现文档块缓存，正在加载...")
            with open(self.chunks_cache, 'rb') as f:
                self.doc_chunks = pickle.load(f)
            print(f"从缓存加载了 {len(self.doc_chunks)} 个文档块")
            return
        
        self.doc_chunks = []
        
        for doc in self.documents:
            chunks = self.split_text_into_chunks(doc['content'])
            for i, chunk in enumerate(chunks):
                self.doc_chunks.append({
                    'content': chunk,
                    'source': doc['filename'],
                    'chunk_id': i,
                    'full_path': doc['path']
                })
        
        # 保存缓存
        with open(self.chunks_cache, 'wb') as f:
            pickle.dump(self.doc_chunks, f)
        
        print(f"文档预处理完成，共生成 {len(self.doc_chunks)} 个文档块")
    
    def load_stopwords(self) -> set:
        """加载中文停用词库"""
        stopwords_file = Path("中文停用词库.txt")
        stopwords = set()
        
        if stopwords_file.exists():
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            stopwords.add(word)
                print(f"已加载 {len(stopwords)} 个停用词")
            except Exception as e:
                print(f"加载停用词库时出错: {e}")
        else:
            print("未找到停用词库文件，将使用默认停用词")
            # 添加一些基本的停用词
            stopwords.update(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
        
        return stopwords
    
    def chinese_tokenizer(self, text):
        """中文分词器（带停用词过滤）"""
        # 使用jieba进行分词
        words = list(jieba.cut(text))
        
        # 过滤停用词、标点符号和空白字符
        filtered_words = []
        punctuation = '，。！？；：""（）【】《》、'
        
        for word in words:
            word = word.strip()
            if (word and 
                len(word) > 1 and  # 过滤单字符（除了一些有意义的单字）
                word not in self.stopwords and 
                not word.isspace() and 
                not all(char in punctuation for char in word)):
                filtered_words.append(word)
        
        return filtered_words
    
    def build_vector_index(self) -> None:
        """构建向量索引"""
        print("正在构建向量索引...")
        
        # 检查缓存
        if self.vectors_cache.exists():
            print("发现向量缓存，正在加载...")
            with open(self.vectors_cache, 'rb') as f:
                cache_data = pickle.load(f)
                self.vectorizer = TfidfVectorizer(
                    tokenizer=self.chinese_tokenizer,
                    vocabulary=cache_data['vocabulary']
                )
                self.vectorizer.idf_ = cache_data['idf'] # 设置idf_
                # TfidfVectorizer 的 _tfidf (TfidfTransformer) 也需要被正确配置
                # 设置 idf_ 应该会通过属性设置器处理内部的 TfidfTransformer
                # 同时，确保 TfidfVectorizer 内部的 _tfidf 对象也认为自己是 fitted
                # 通常，TfidfTransformer 检查 _idf_diag 是否存在。设置 idf_ 会间接处理这个。

                self.doc_vectors = cache_data['vectors']
            print("向量索引加载完成")
            return
        
        # 提取文档块内容
        texts = [chunk['content'] for chunk in self.doc_chunks]
        
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.chinese_tokenizer,
            lowercase=False,
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # 构建向量
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
        # 保存缓存 - 保存词汇表、向量和idf_
        cache_data = {
            'vocabulary': self.vectorizer.vocabulary_,
            'vectors': self.doc_vectors,
            'idf': self.vectorizer.idf_  # 保存idf_
        }
        with open(self.vectors_cache, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print("向量索引构建完成")
    
    def search_relevant_chunks(self, query: str, top_k: int = 10, similarity_threshold: float = 0.01) -> List[Dict]:
        """搜索相关文档块
        
        Args:
            query: 查询文本
            top_k: 最大返回文档块数量
            similarity_threshold: 相似度阈值，只返回相似度高于此值的结果
        """
        if self.vectorizer is None or self.doc_vectors is None:
            raise ValueError("向量索引未构建，请先调用 build_vector_index()")
        
        # 向量化查询
        query_vector = self.vectorizer.transform([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # 获取所有相似度高于阈值的文档块
        valid_indices = np.where(similarities > similarity_threshold)[0]
        
        # 按相似度排序
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # 限制返回数量
        top_indices = sorted_indices[:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.doc_chunks[idx].copy()
            chunk['similarity'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """使用DeepSeek API生成答案"""
        # 构建上下文
        context = "\n\n".join([f"文档片段{i+1}：{chunk['content']}" 
                              for i, chunk in enumerate(context_chunks)])
        # 我测试了多个版本。可以换一下提示词对比哪个效果好，目前这个我觉得比较好
        system_prompt = """
# Role: 红楼梦研究专家

## Profile
- language: 中文
- description: 精通《红楼梦》文本及红学研究，能够深入解析作品的人物、情节、诗词及文化内涵
- background: 多年从事《红楼梦》研究与教学，参与过多项红学课题研究
- personality: 严谨细致，富有文人气质，善于引经据典
- expertise: 文本分析、人物研究、诗词鉴赏、文化解读
- target_audience: 红学爱好者、文学研究者、学生群体

## Skills

1. 文本解析能力
   - 情节梳理: 能准确还原小说情节脉络
   - 细节把握: 对文本细节有敏锐洞察力
   - 人物分析: 深入剖析人物性格与命运
   - 诗词解读: 精准解析书中诗词内涵

2. 学术研究能力
   - 文献考证: 熟悉各类红学研究成果
   - 文化阐释: 揭示作品背后的文化内涵
   - 比较研究: 能与其他文学作品进行对比
   - 版本鉴别: 了解不同版本差异

## Rules

1. 回答原则：
   - 基于文本: 所有回答必须严格依据原著文本
   - 严谨准确: 不妄加猜测，不传播未经考证的观点
   - 深度解析: 透过表面现象揭示深层含义
   - 客观公正: 避免个人主观臆断

2. 行为准则：
   - 引经据典: 重要观点需引用原文佐证
   - 语言典雅: 保持与原著相称的文雅风格
   - 层次分明: 回答要有逻辑性和条理性
   - 深入浅出: 复杂问题要解释得通俗易懂

3. 限制条件：
   - 不涉争议: 避免介入红学争议性话题
   - 不妄评续作: 对后四十回保持审慎态度
   - 不越文本: 不脱离文本过度解读
   - 不代作者: 不以作者口吻发表观点

## Workflows

- 目标: 提供专业准确的红楼梦解读
- 步骤 1: 仔细理解用户问题，明确询问重点
- 步骤 2: 检索相关文本段落，确认信息准确性
- 步骤 3: 组织回答内容，适当引用原文
- 步骤 4: 以典雅文风呈现完整解答
- 预期结果: 用户获得权威、深入的红楼梦知识

## Initialization
作为红楼梦研究专家，你必须遵守上述Rules，按照Workflows执行任务。
        """.strip()
        
        user_prompt = f"""
基于以下文档内容回答问题：

{context}

问题：{query}
        """.strip()
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_type = type(e).__name__
            error_details = str(e) if str(e) else "未知错误"
            print(f"生成答案时出错: {error_type}: {error_details}")
            
            # 确保返回有效的错误信息字符串
            return f"生成答案时出错: {error_type} - {error_details}"
    
    def ask(self, question: str, top_k: int = 10, similarity_threshold: float = 0.01) -> Dict:
        """问答接口
        
        Args:
            question: 用户问题
            top_k: 最大返回文档块数量，默认10
            similarity_threshold: 相似度阈值，默认0.01
        """
        print(f"\n问题: {question}")
        print("正在搜索相关内容...")
        
        # 搜索相关文档块
        relevant_chunks = self.search_relevant_chunks(question, top_k=top_k, similarity_threshold=similarity_threshold)
        
        if not relevant_chunks:
            return {
                'question': question,
                'answer': '抱歉，在文档中没有找到与您问题相关的内容。',
                'sources': []
            }
        
        print(f"找到 {len(relevant_chunks)} 个相关文档片段")
        
        # 生成答案
        print("正在生成答案...")
        answer = self.generate_answer(question, relevant_chunks)
        
        # 整理来源信息
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                'source': chunk['source'],
                'similarity': chunk['similarity'],
                'content_preview': chunk['content'][:100] + '...' if len(chunk['content']) > 100 else chunk['content']
            })
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources
        }
    
    def initialize(self) -> None:
        """初始化系统"""
        print("=== 红楼梦RAG问答系统初始化 ===")
        self.load_documents()
        if not self.documents:
            raise ValueError("未找到任何文档，请检查docs目录")
        
        self.preprocess_documents()
        self.build_vector_index()
        print("系统初始化完成！\n")

def main():
    """主函数"""
    # 加载.env文件
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("请设置环境变量 DEEPSEEK_API_KEY")
        print("或者直接在代码中设置API密钥")
        # 如果没有环境变量，可以在这里直接设置
        # api_key = ""
        return
    
    try:
        # 初始化RAG系统
        rag = RedMansionRAG(api_key=api_key)
        rag.initialize()
        
        # 交互式问答
        print("=== 红楼梦智能问答系统 ===")
        print("输入您的问题，输入 'quit' 或 'exit' 退出")
        print("示例问题：")
        print("- 甄士隐是谁？")
        print("- 贾雨村的故事是什么？")
        print("- 通灵宝玉是什么？")
        print("- 英莲发生了什么事？")
        print()
        
        while True:
            question = input("请输入您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用红楼梦问答系统！")
                break
            
            if not question:
                continue
            
            # 获取答案
            result = rag.ask(question)
            
            # 显示结果
            print(f"\n答案: {result['answer']}")
            
            if result['sources']:
                print("\n相关文档片段:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. 来源: {source['source']} (相似度: {source['similarity']:.3f})")
                    print(f"   内容预览: {source['content_preview']}")
            
            print("\n" + "="*50 + "\n")
    
    except Exception as e:
        print(f"系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()