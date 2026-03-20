from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model_type = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides information about {topic}."),
    ("human", "can you tell me something about {topic}?"),
])

chat_prompt2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant that provides information about {topic}."),
    HumanMessagePromptTemplate.from_template("Can you tell me something interesting about {topic}.")
])

prompt_text = chat_prompt.format_messages({"topic" : "Binomial expansion"})
prompt_text_2 = chat_prompt2.format_messages({"topic" : "Politics"})

response_1 = model_type.invoke(prompt_text.content)
response_2 = model_type.invoke(prompt_text_2.content)


print(response_1)
print("\n------------------------------------------------\n")
print(response_2)