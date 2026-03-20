from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.4
)

response = model.invoke("Tell me a secret about successful business men in 100 words.")
print(response.content)