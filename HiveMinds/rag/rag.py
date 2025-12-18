import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import json
from pathlib import Path
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatOpenAI

from HiveMinds.engine.factory import create_llm_engine
from HiveMinds.context.build_context import build_retrieval_prompt

class PDE_RAG:
    def __init__(self, embedding_model_name, rag_root_path=None, top_k=2, model_info=None):
        self.rag_root_path = rag_root_path if rag_root_path else os.path.join(os.path.dirname(__file__), "database")
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.model_info = model_info if model_info else "deepseek-chat"
        
        self.rag_interface = self.load_rag_interface()
        
    def load_rag_interface(self):
        self.pde_problem_vectordb = FAISS.load_local(
            os.path.join(self.rag_root_path, "pde_problem_faiss"), 
            HuggingFaceEmbeddings(model_name=self.embedding_model_name),
            allow_dangerous_deserialization=True)
        self.pde_case_vectordb = FAISS.load_local(
            os.path.join(self.rag_root_path, "pde_case_faiss"), 
            HuggingFaceEmbeddings(model_name=self.embedding_model_name),
            allow_dangerous_deserialization=True)
        # chat_model = create_llm_engine(self.model_info)
        chat_model = ChatDeepSeek(model='deepseek-chat', temperature=0.0)
        retriever = self.pde_problem_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k})
        qa_interface = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_interface
        
    def get_rag_case(self, query):
        prompt_retrieval = build_retrieval_prompt(query)
        rsp = self.rag_interface(prompt_retrieval)
        retrieval_doc = rsp['source_documents'][0]  # Document 对象
        retrieval_pde_case = self.pde_case_vectordb.similarity_search(query="", k=1 ,filter={"uid": retrieval_doc.metadata['uid']})[0].page_content
        retrieval_pde_case = json.loads(retrieval_pde_case)
        return retrieval_pde_case
    
    def __call__(self, query):
        return self.get_rag_case(query)
        
        