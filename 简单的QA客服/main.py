from langchain.text_splitter import RecursiveCharacterTextSplitter                                  
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
import os
import openai

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['OPENAI_API_KEY'] = 'your_openai_key'                
os.environ['OPENAI_API_BASE'] = ''


def run():
    loader = TextLoader('qa.txt')
    data = loader.load()
 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 50, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)                                                
                                                                                                    
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())         
                                                                                                
    while True:                                                                                     
        print("请输入问题，例如：弦丝画制作的活动时长是多少？")                                     
        query = input("请输入:")                                                                    
        if "end" == query:                                                                          
            break                                                                                   
        question = query                                                                            
        print(f"问题：{question}")                                                                  
                                                                                                   
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)                                 
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())           
        ret = qa_chain({"query": question})                                                         
        print(f"回答：{ret['result']}")                                                             
 if '__main__' == __name__:                                                                          
     run()
