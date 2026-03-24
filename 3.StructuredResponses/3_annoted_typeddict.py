from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature = 0.7
)

# Schema for Structured response
class Schema(TypedDict):
    key_themes: Annotated[list[str], "Must write down all the key themes discussed in the review in a list."]
    summary: Annotated[str, "Must write a concise summary of the review in one sentence."]
    sentiment: Annotated[str, "Must write overall sentiment of the review as either Positive, Negative or Neutral."]
    pros: Annotated[Optional[list[str]], "Must write down the pros mentioned in the review in a list. If no pros are mentioned, return an empty list."]
    cons: Annotated[Optional[list[str]], "Must write down the cons mentioned in the review in a list. If no cons are mentioned, return an empty list"]


structured_model = model.with_structured_output(Schema)

# Prompt to generate structured response
prompt = """
The Infinix zero 30 offers impressive value for it's price, combining stylish design with solid everyday perfromance. It's large AMOLED display is bright, colorful and smooth, making it great for streaming videos, browsing social media and casual gaming.The phone feels sleek in hand, with a modern finish that gives it a premuim look despite being budget-friendly.

Performance is reliable for daily tasks like messaging, multitasking, and light gaming, with minimal lag. The camera setup captures detailed photos in good lighting conditions, and the front camera is particularly strong for selfies and video calls. Battery life easily lasts a full day on moderate use, and fast charging support helps you power up quickly when needed. Overall, it’s a well-rounded smartphone that delivers strong features and performance at an affordable price point.
"""
response = structured_model.invoke(prompt)

print(response) 


# Final output will be a structured response adhering to the Review TypedDict schema, making it easier to parse and utilize the information in downstream applications.

# Something like this will be the output:
{'cons': [], 'key_themes': ['value for money', 'design', 'performance', 'camera', 'battery life'], 'pros': ['stylish design', 'solid everyday performance', 'large AMOLED display', 'reliable camera', 'long-lasting battery life'], 'sentiment': 'Positive', 'summary': 'The Infinix Zero 30 offers great value for its price, with a stylish design, solid everyday performance, and a range of strong features, including a large AMOLED display, reliable camera, and long-lasting battery life.'}