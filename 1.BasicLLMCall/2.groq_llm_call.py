from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model = "llama-3.3-70b-versatile", temperature = 0)

response = model.invoke("Tell me a secret about artificial intelligence in 100 words.")
print(response.content)