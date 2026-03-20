from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider = "groq",
    temperature = 0
)
static_prompt = PromptTemplate.from_template("Tell me about mathematical graphs, the different types and a cheat to understanding and solving them.")
prompt_text = static_prompt.format()

response = model.invoke(prompt_text)
print(response.content)