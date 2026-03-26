from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = init_chat_model(
    model= "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.7
)

parser = JsonOutputParser()

# You must provide the model with format instructions so it knows how to format its output correctly for the parser to understand. The get_format_instructions method of the JsonOutputParser provides these instructions in a way that can be easily included in the prompt.
template = PromptTemplate.from_template(
    """Give me the name, age and city of a fictional person. 
    Also the name and city has to be Nigerian.\n{format_instructions}
    """
)

template_with_format = template.partial(format_instructions=parser.get_format_instructions())

chain = template_with_format | model | parser

response = chain.invoke({})

print(response)
