from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing import Literal , Optional
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.7
)

# Schema for structured response
class Schema(BaseModel):
    key_themes: list[str] = Field(..., description="Must write down all the key themes discussed in the review in a list.")
    summary: str = Field(..., description="Must write a concise summary of the review in one sentences.")
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(..., description="Must write the overall sentiment of the review as either Positive, Negative, or Neutral.")
    pros: Optional[list[str]] = Field(default=None, description="Must write down the pros mentioned in the review in a list. If no pros are mentioned, return null.")
    cons: Optional[list[str]] = Field(default=None, description="Must write down the cons mentioned in the review in a list. If no cons are mentioned, return null.")
    name: Optional[str] = Field(default=None, description="Must write down the name of the reviewer if mentioned in the review. If no name is mentioned, return null.")

structured_model = model.with_structured_output(Schema , strict=True) # strict=True ensures that the model adheres strictly to the defined schema, and will raise an error if the response does not conform to the schema. This is useful for ensuring data integrity and consistency in the structured output.


# Prompt to generate structured response
prompt =  """
Hey name is Martin Uba. The Infinix Zero 30 offers impressive value for its price, combining stylish design with solid everyday performance. Its large AMOLED display is bright, colorful, and smooth, making it great for streaming videos, browsing social media, and casual gaming. The phone feels sleek in hand, with a modern finish that gives it a premium look despite being budget-friendly.

Performance is reliable for daily tasks like messaging, multitasking, and light gaming, with minimal lag. The camera setup captures detailed photos in good lighting conditions, and the front camera is particularly strong for selfies and video calls. Battery life easily lasts a full day on moderate use, and fast charging support helps you power up quickly when needed. Overall, it’s a well-rounded smartphone that delivers strong features and performance at an affordable price point.
"""

response = structured_model.invoke(prompt)

print(response) # This is the end of the code snippet for this file.


# Final output will be a structured response adhering to the Review TypedDict schema, making it easier to parse and utilize the information in downstream applications.

# Something like this will be the output:
# key_themes=['design', 'performance', 'display', 'camera', 'battery life'] summary='The Infinix Zero 30 offers impressive value for its price with its stylish design, solid everyday performance, and well-rounded features' sentiment='Positive' pros=['stylish design', 'solid performance', 'large AMOLED display', 'detailed camera photos', 'long battery life'] cons=None name='Macsmith Okorie'