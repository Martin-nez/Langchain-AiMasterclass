from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider = "groq",
    temperature = 0.8
)

# Prompt to generate structured response
prompt = """
The hardware is great but software feels bloated. There are too many pre-installed apps that i never use, and it slows down the device. However, the battery life is interesting and the display is stunning. Hoping for a software update to improve performance. Overall, it's a mixed experience.
"""
response = model.invoke(prompt)
print(type(prompt))
print("\n---------------------------------------------------\n")
print(response.content)