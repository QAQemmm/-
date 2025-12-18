# %%
import os
from typing import TypedDict, List, Optional, Literal, Dict, Any
from typing_extensions import Annotated
import operator
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# LangGraph和LangChain相关
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 千问API
import dashscope
from dashscope import Generation

# 设置API密钥
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key")

class AgentState(TypedDict):
    """Agent状态定义"""
    messages: Annotated[List[Dict], operator.add]  # 对话历史
    question: str  # 当前问题
    needs_retrieval: Optional[bool] = None  # 是否需要检索
    retrieved_docs: List[Document]  # 检索到的文档
    context: str  # 上下文信息
    response: str  # AI响应

class QwenAgent:
    """千问Agent实现"""
    
    def __init__(self, model_name: str = "qwen-max", temperature: float = 0.7):
        """
        初始化Agent
        
        Args:
            model_name: 千问模型名称
            temperature: 温度参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.vector_store = None
        self.embeddings = None
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        """初始化嵌入模型"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    
    def create_vector_store(self, documents: List[str]):
        """
        创建向量存储
        
        Args:
            documents: 文档列表
        """
        # 文档分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        # 创建文档对象
        docs = []
        for i, text in enumerate(documents):
            docs.append(Document(
                page_content=text,
                metadata={"source": f"doc_{i}", "index": i}
            ))
        
        # 分割文档
        splits = text_splitter.split_documents(docs)
        
        # 创建向量存储
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def load_existing_vector_store(self):
        """加载已有的向量存储"""
        if os.path.exists("./chroma_db"):
            self.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )
    
    def judge_retrieval_needed(self, question: str) -> bool:
        """
        判断是否需要检索
        
        Args:
            question: 用户问题
            
        Returns:
            bool: 是否需要检索
        """
        # 需要检索的关键词类型
        retrieval_keywords = [
            "文档", "文件", "资料", "内容", "信息",
            "查询", "查找", "搜索", "检索",
            "根据", "依据", "参考", "按照"
        ]
        
        # 具体事实性问题
        factual_keywords = [
            "什么", "谁", "哪里", "何时", "为什么", "如何",
            "多少", "哪些", "是不是", "有没有"
        ]
        
        question_lower = question.lower()
        
        # 如果有具体的事实性询问关键词，且包含检索关键词，则可能需要检索
        has_factual = any(keyword in question for keyword in factual_keywords)
        has_retrieval = any(keyword in question_lower for keyword in retrieval_keywords)
        
        # 简单规则：如果是事实性问题且涉及文档内容，则需要检索
        if has_factual and has_retrieval:
            return True
        
        # 也可以用模型判断（更准确）
        return self._llm_judge_retrieval(question)
    
    def _llm_judge_retrieval(self, question: str) -> bool:
        """
        使用LLM判断是否需要检索
        
        Args:
            question: 用户问题
            
        Returns:
            bool: 是否需要检索
        """
        prompt = f"""请判断以下问题是否需要检索文档内容来回答：
        
        问题：{question}
        
        请分析：
        1. 这个问题是否需要查找特定文档、数据或信息？
        2. 这个问题是否能基于一般知识回答？
        3. 这个问题是否涉及具体的、非公开的信息？
        
        请只回答"需要检索"或"不需要检索"。
        """
        
        try:
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.output.text.strip()
            return "需要检索" in result
        except:
            # 如果API调用失败，回退到规则判断
            return False
    
    def retrieve_documents(self, question: str, k: int = 3) -> List[Document]:
        """
        检索相关文档
        
        Args:
            question: 用户问题
            k: 返回的文档数量
            
        Returns:
            List[Document]: 相关文档列表
        """
        if not self.vector_store:
            return []
        
        # 相似度检索
        docs = self.vector_store.similarity_search(question, k=k)
        return docs
    
    def call_qwen(self, messages: List[Dict], context: str = "") -> str:
        """
        调用千问API
        
        Args:
            messages: 消息历史
            context: 检索到的上下文
            
        Returns:
            str: AI响应
        """
        # 构建系统提示
        system_prompt = """你是一个智能助手，负责回答用户的问题。
        
        回答要求：
        1. 如果提供了上下文信息，请基于上下文回答
        2. 如果上下文不包含相关信息，请基于你的知识回答
        3. 保持回答准确、简洁、有帮助
        """
        
        # 如果有上下文，添加到系统提示中
        if context:
            system_prompt += f"\n\n上下文信息：\n{context}\n\n请基于以上上下文回答问题。"
        
        # 构建完整的消息列表
        full_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        
        try:
            # 调用千问API
            response = Generation.call(
                model=self.model_name,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=2000
            )
            
            return response.output.text
        except Exception as e:
            print(f"调用千问API失败: {e}")
            return "抱歉，我暂时无法回答这个问题。请稍后再试。"

class RAGAgent:
    """RAG Agent主类"""
    
    def __init__(self):
        """初始化RAG Agent"""
        self.agent = QwenAgent()
        self.graph = self.build_graph()
        self.checkpointer = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}
        
        # 加载或创建向量存储
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """初始化知识库"""
        # 尝试加载已有的向量存储
        self.agent.load_existing_vector_store()
        
        # 如果没有，可以在这里添加默认文档
        if not self.agent.vector_store:
            default_docs = [
                "LangGraph是一个用于构建多步、有状态AI应用的框架。",
                "RAG（检索增强生成）结合了检索系统和生成模型。",
                "千问Max是阿里巴巴开发的大型语言模型。",
                "多轮对话需要维护对话历史和上下文信息。"
            ]
            self.agent.create_vector_store(default_docs)
            print("知识库已使用默认文档初始化")
    
    def add_documents(self, documents: List[str]):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
        """
        self.agent.create_vector_store(documents)
        print(f"已添加 {len(documents)} 个文档到知识库")
    
    def judge_retrieval_node(self, state: AgentState) -> AgentState:
        """
        判断是否需要检索的节点
        
        Args:
            state: 当前状态
            
        Returns:
            AgentState: 更新后的状态
        """
        question = state["question"]
        needs_retrieval = self.agent.judge_retrieval_needed(question)
        
        state["needs_retrieval"] = needs_retrieval
        print(f"问题: '{question}' - 需要检索: {needs_retrieval}")
        
        return state
    
    def retrieval_node(self, state: AgentState) -> AgentState:
        """
        检索节点
        
        Args:
            state: 当前状态
            
        Returns:
            AgentState: 更新后的状态
        """
        question = state["question"]
        retrieved_docs = self.agent.retrieve_documents(question)
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        state["retrieved_docs"] = retrieved_docs
        state["context"] = context
        
        print(f"检索到 {len(retrieved_docs)} 个相关文档")
        
        return state
    
    def generate_response_node(self, state: AgentState) -> AgentState:
        """
        生成响应节点
        
        Args:
            state: 当前状态
            
        Returns:
            AgentState: 更新后的状态
        """
        # 获取对话历史
        messages = state.get("messages", [])
        question = state["question"]
        
        # 将用户问题添加到消息历史
        messages.append({"role": "user", "content": question})
        
        # 获取上下文
        context = state.get("context", "")
        
        # 调用千问生成响应
        response = self.agent.call_qwen(messages, context)
        
        # 将AI响应添加到消息历史
        messages.append({"role": "assistant", "content": response})
        
        state["response"] = response
        state["messages"] = messages
        
        print(f"生成响应完成")
        
        return state
    
    def direct_response_node(self, state: AgentState) -> AgentState:
        """
        直接响应节点（无需检索）
        
        Args:
            state: 当前状态
            
        Returns:
            AgentState: 更新后的状态
        """
        # 没有检索，直接生成响应
        state["context"] = ""
        state["retrieved_docs"] = []
        
        return self.generate_response_node(state)
    
    def route_based_on_retrieval(self, state: AgentState) -> str:
        """
        根据是否需要检索路由到不同节点
        
        Args:
            state: 当前状态
            
        Returns:
            str: 下一个节点名称
        """
        if state["needs_retrieval"]:
            return "retrieval"
        else:
            return "direct_response"
    
    def build_graph(self):
        """
        构建工作流图
        
        Returns:
            StateGraph: 构建好的图
        """
        # 创建图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("judge_retrieval", self.judge_retrieval_node)
        workflow.add_node("retrieval", self.retrieval_node)
        workflow.add_node("direct_response", self.direct_response_node)
        workflow.add_node("generate_response", self.generate_response_node)
        
        # 设置入口点
        workflow.set_entry_point("judge_retrieval")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "judge_retrieval",
            self.route_based_on_retrieval,
            {
                "retrieval": "retrieval",
                "direct_response": "direct_response"
            }
        )
        
        # 添加普通边
        workflow.add_edge("retrieval", "generate_response")
        workflow.add_edge("direct_response", END)
        workflow.add_edge("generate_response", END)
        
        # 编译图
        return workflow.compile()
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            question: 用户问题
            
        Returns:
            Dict: 处理结果
        """
        # 初始状态
        initial_state = AgentState(
            messages=[],
            question=question,
            needs_retrieval=None,
            retrieved_docs=[],
            context="",
            response=""
        )
        
        # 执行图
        result = self.graph.invoke(
            initial_state,
            config=self.config
        )
        
        return {
            "response": result["response"],
            "needs_retrieval": result["needs_retrieval"],
            "retrieved_docs_count": len(result["retrieved_docs"]),
            "context": result["context"],
            "messages": result["messages"]
        }
    
    def chat(self):
        """交互式聊天"""
        print("=" * 50)
        print("RAG Agent 聊天系统")
        print("支持多轮对话，输入 'quit' 或 '退出' 结束")
        print("=" * 50)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n你: ").strip()
                
                if user_input.lower() in ['quit', '退出', 'exit']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                # 处理查询
                result = self.process_query(user_input)
                
                # 显示响应
                print(f"\n助手: {result['response']}")
                
                # 显示检索信息（可选）
                if result['needs_retrieval'] and result['retrieved_docs_count'] > 0:
                    print(f"[已检索 {result['retrieved_docs_count']} 个相关文档]")
                
            except KeyboardInterrupt:
                print("\n\n对话已中断")
                break
            except Exception as e:
                print(f"错误: {e}")
                continue

def main():
    """主函数"""
    # 创建Agent
    rag_agent = RAGAgent()
    
    # 可选：添加自定义文档
    custom_docs = input("是否要添加自定义文档？(y/n): ").strip().lower()
    if custom_docs == 'y':
        docs = []
        print("请输入文档内容（输入空行结束）：")
        while True:
            line = input()
            if not line.strip():
                break
            docs.append(line)
        
        if docs:
            rag_agent.add_documents(docs)
    
    # 开始聊天
    rag_agent.chat()

if __name__ == "__main__":
    main()


