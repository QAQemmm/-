# %%
# example_usage.py
from rag_agent import RAGAgent
from document_loader import load_custom_documents

def main():
    # 创建Agent
    agent = RAGAgent()
    
    # 加载自定义文档
    load_custom_documents(agent, "./my_documents")
    
    # 单次查询示例
    queries = [
        "LangGraph是什么？",
        "给我讲个笑话",
        "RAG有什么优势？",
        "今天的天气怎么样？",
        "文档中提到的多轮对话是什么意思？"
    ]
    
    for query in queries:
        print(f"\n用户: {query}")
        result = agent.process_query(query)
        print(f"助手: {result['response'][:100]}...")
        print(f"需要检索: {result['needs_retrieval']}")
        print("-" * 50)
    
    # 或者启动交互式聊天
    # agent.chat()

if __name__ == "__main__":
    main()


