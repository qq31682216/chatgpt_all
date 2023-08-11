from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import os
import openai

os.environ["OPENAI_API_KEY"] = 'your_openai_key'
os.environ['OPENAI_API_BASE'] = ''

llm = ChatOpenAI(temperature=0)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)
tools = [get_word_length]
system_message = SystemMessage(content="你是非常强大的助手，但不善于计算单词的长度。")

MEMORY_KEY = "chat_history"
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
)

memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
agent_executor.run("耕耘一词中有几个字？")
agent_executor.run("它是个真正的词汇吗？")
