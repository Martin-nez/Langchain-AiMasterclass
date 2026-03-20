from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
#from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model_type = ChatGroq(
    model="llama3-70b-8192",
    model_provider="groq",
    temperature=0
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides information about {topic}."),
    ("human", "Can you tell me something about {topic}?"),
])

chat_prompt2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant that provides information about {topic}."),
    HumanMessagePromptTemplate.from_template("Can you tell me something interesting about {topic}.")
])

prompt_text = chat_prompt.format_messages(topic = "Binomial expansion")
prompt_text_2 = chat_prompt2.format_messages(topic = "Politics")

response_1 = model_type.invoke(prompt_text)
response_2 = model_type.invoke(prompt_text_2)

print(response_1.content)
print("\n------------------------------------------------\n")
print(response_2.content)


# import requests
# print(requests.get("https://api.groq.com").status_code)


# import os
# print("API KEY:", os.getenv("GROQ_API_KEY"))

#C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe -m pip install --upgrade pip 