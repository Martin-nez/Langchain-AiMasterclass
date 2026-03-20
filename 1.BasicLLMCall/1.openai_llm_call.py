from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-5-nano", temperature = 0)

model_response = model.invoke("Tell me a joke about artificial intelligence in 50 words.")
print(model_response.content)
