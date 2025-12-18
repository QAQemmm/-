# -
深度赋智笔试项目-林

📦 快速开始

安装依赖
bash
pip install -r requirements.txt
环境配置
复制环境变量文件：

bash
cp .env.example .env
编辑 .env 文件，设置您的 API 密钥：

text
DASHSCOPE_API_KEY=your-api-key-here
MODEL_NAME=qwen-max
TEMPERATURE=0.7
运行示例
python
python rag_agent.py
python example_usage.py
📚 使用方式
1. 添加文档到知识库
python
from rag_agent import RAGAgent

agent = RAGAgent()

documents = [
    "LangGraph 是一个用于构建多步骤、有状态 AI 应用的框架。",
    "RAG（检索增强生成）结合了检索系统和生成模型。"
]
agent.add_documents(documents)
2. 批量加载文档
将文档放入 documents/ 目录，支持格式：

📄 PDF (.pdf)

📝 Word (.docx)

📋 纯文本 (.txt)

📓 Markdown (.md)

python
from document_loader import load_custom_documents

load_custom_documents(agent, "./my_documents")
3. 交互式对话
启动后，您可以：

输入普通问题：基于 AI 知识回答

输入需要检索的问题：基于文档内容回答

输入 quit 或 退出：结束对话


python
class AgentState:
    messages: List[Dict]      # 对话历史
    question: str            # 当前问题
    needs_retrieval: bool    # 是否需要检索
    retrieved_docs: List     # 检索到的文档
    context: str            # 上下文信息
    response: str           # AI 响应
