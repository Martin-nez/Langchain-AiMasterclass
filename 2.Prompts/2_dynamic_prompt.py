from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature =0
)
dynamic_prompt = PromptTemplate.from_template("Write a short paragraph about {topic} in an {style} style.")

prompt_text = dynamic_prompt.format(topic="AI", style = "informative") # we can change the topic and style dynamically when formatting the prompt

response = model.invoke(prompt_text)
print(response.content)