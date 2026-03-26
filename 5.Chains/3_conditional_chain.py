from langchain.chat_models import init_chat_model
from typing import Literal
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

# Initialiize groq model (langchain 1.x style)
model = init_chat_model(
    model= "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.7
)

class FeedBack(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        ..., description="The sentiment for the feedback"
    )

parser_pydantic = PydanticOutputParser(pydantic_object=FeedBack)
parser_str = StrOutputParser()


classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment classifier."),
    ("human", 
     "Classify the sentiment of the following text into Positive or Negative. {feedback}\n {format_instructions}"
    )
])

classify_prompt_with_format =   classify_prompt.partial(
        format_instructions=parser_pydantic.get_format_instructions()
    ) # We use this partial method to inject the format instructions from the PydanticOutputParser into the prompt. This ensures that the model knows how to format its output correctly for the parser to understand.


classify_chain = classify_prompt_with_format | model | parser_pydantic
    

positive_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write an appropriate response for this positive feedback:\n{feedback}")
])

negative_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write an appropriate response for this negative feedback:\n{feedback}")
])

positive_chain = positive_prompt | model | parser_str
negative_chain = negative_prompt | model | parser_str


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", positive_chain),
    (lambda x: x.sentiment == "Negative", negative_chain),
    RunnableLambda(lambda x: "No valid sentiment found"), # This is the default
)


full_chain = classify_chain | branch_chain

response = full_chain.invoke({
    "feedback": "I love the new design of your website!" # Try changing this feedback to a positive one to see how the response changes.
})

print(response)


#         +-------------------+
#         |  sentiment == ?   |
#         +-------------------+
#            /        |        \
#           /         |         \
#          v          v          v
# +----------------+ +----------------+ +--------------------------+
# |  "Positive"    | |  "Negative"    | |  Default (No Sentiment) |
# +----------------+ +----------------+ +--------------------------+
#          |                 |                     |
#          v                 v                     v
# +----------------+  +----------------+   +-----------------------+
# | Positive Chain |  | Negative Chain |   |   RunnableLambda      |
# |----------------|  |----------------|   | "No valid sentiment"  |
# | Prompt -> LLM  |  | Prompt -> LLM  |   +-----------------------+
# | -> StrParser   |  | -> StrParser   |
# +----------------+  +----------------+
#          \                 /
#           \               /
#            v             v
#         +---------------------+
#         |    Final Response   |
#         +---------------------+
#                   |
#                   v
#         +---------------------+
#         |    print(response)  |
#         +---------------------+