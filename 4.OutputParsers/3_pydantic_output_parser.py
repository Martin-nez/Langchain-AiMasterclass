from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


model = init_chat_model(
    model= "lla ma-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.7
)

class Person(BaseModel):
    name: str = Field(..., description="The person's full name")
    age: int = Field(gt=18 , lt=60, description="The age of the person. Must not be lesser than 18 or greater the 60") # greater than 18 and lesser than 60
    city: str = Field(..., description="The city where the person lives in.")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate.from_template(
    """Give me the name, age and city of a fictional {place} person.
    \n{format_instructions}
    """
)

template_with_format = template.partial(format_instructions=parser.get_format_instructions())

chain = template_with_format | model | parser

response = chain.invoke({"place":"France"})

print(response)

# We always add format_instructions in the prompt when dealing with structured output parsers like PydanticOutputParser or JsonOutputParser. This ensures the model knows how to format its output correctly for the parser to understand.