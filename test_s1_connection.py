from langchain_community.llms import OpenAI
import os

PANSHI_API_BASE = "https://uni-api.cstcloud.cn/v1"  
PANSHI_API_KEY = os.getenv("PANSHI_API_KEY")      

from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    api_key=PANSHI_API_KEY,
    base_url=PANSHI_API_BASE,
    model="S1-Base-Lite",
    temperature=0.2,
    timeout=60,
)

resp = llm.invoke("What is the weather in Beijing")
print(resp.content)