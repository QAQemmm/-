# %%
# config.py
import os
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Agent配置"""
    # 模型配置
    MODEL_NAME: str = "qwen-max"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # RAG配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_K: int = 3  # 检索文档数量
    
    # 向量数据库配置
    PERSIST_DIRECTORY: str = "./chroma_db"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 路径配置
    DOCUMENTS_DIR: str = "./documents"
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            MODEL_NAME=os.getenv("MODEL_NAME", cls.MODEL_NAME),
            TEMPERATURE=float(os.getenv("TEMPERATURE", cls.TEMPERATURE)),
            MAX_TOKENS=int(os.getenv("MAX_TOKENS", cls.MAX_TOKENS)),
        )

# 环境变量示例 .env
"""
DASHSCOPE_API_KEY=your-api-key-here
MODEL_NAME=qwen-max
TEMPERATURE=0.7
MAX_TOKENS=2000
"""


