from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AIMessage,HumanMessage,SystemMessage
from langchain.chat_models import ChatOpenAI
import os
import openai
os.environ['OPENAI_API_KEY'] = 'your_openai_key'
os.environ['OPENAI_API_BASE'] = ''
chat = ChatOpenAI()
chat([HumanMessage(content="Translate this sentence from English to French: I love programming.")])

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

llm = ChatOpenAI()
prompt = ChatPromptTemplate(messages=[SystemMessagePromptTemplate.from_template("You are a nice chatbot having a conversation with a human."),MessagesPlaceholder(variable_name="chat_history"),HumanMessagePromptTemplate.from_template("{question}")])
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
conversation = LLMChain(llm=llm,prompt=prompt,verbose=True,memory=memory)

response = conversation({"question": "hi"})
response = conversation({"question": "Translate this sentence from English to French: I love programming."})
response = conversation({"question": "Now translate the sentence to German."})
