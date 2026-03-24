from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature = 0.7
)

class Review(TypedDict):
    summary : str
    sentiment: str

# Prompt to generate structured response
prompt ="""
The hardware is great but software feels bloated. There are too many pre-installed apps that i never use, and it slows down the device. However, the battery life is interesting and the display is stunning. Hoping for a software update to improve performance. Overall, it's a mixed experience.
"""

structured_model = model.with_structured_output(Review)

response = structured_model.invoke(prompt)
print(response)



# Final output will be a structured response adhering to the Review TypedDict schema, making it easier to parse and utilize the information in downstream applications.

# Something like this will be the output:
#{'sentiment': 'neutral', 'summary': 'The device has great hardware, but the software is bloated with too many pre-installed apps, slowing it down. However, the battery life and display are impressive.'}
