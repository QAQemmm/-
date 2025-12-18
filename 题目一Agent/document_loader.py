# %%

import os
from typing import List
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)

class DocumentLoader:
    """文档加载器"""
    
    @staticmethod
    def load_documents_from_directory(directory_path: str) -> List[str]:
        """
        从目录加载文档
        
        Args:
            directory_path: 目录路径
            
        Returns:
            List[str]: 文档内容列表
        """
        supported_extensions = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader,
        }
        
        documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in supported_extensions and os.path.isfile(file_path):
                try:
                    loader = supported_extensions[ext](file_path)
                    loaded_docs = loader.load()
                    
                    for doc in loaded_docs:
                        documents.append(doc.page_content)
                    
                    print(f"已加载: {filename}")
                except Exception as e:
                    print(f"加载文件 {filename} 失败: {e}")
        
        return documents

# 使用示例
def load_custom_documents(agent: RAGAgent, docs_dir: str = "./documents"):
    """加载自定义文档"""
    if os.path.exists(docs_dir):
        loader = DocumentLoader()
        documents = loader.load_documents_from_directory(docs_dir)
        
        if documents:
            agent.add_documents(documents)
            print(f"已从 {docs_dir} 加载 {len(documents)} 个文档")
        else:
            print(f"{docs_dir} 目录中没有支持的文档文件")
    else:
        print(f"目录 {docs_dir} 不存在")


