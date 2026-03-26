from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.7
)

prompt_1 = PromptTemplate.from_template("Generate detailed report on {topic}")
prompt_2 = PromptTemplate.from_template("Generate 3 point summary of the following {text}")

parser = StrOutputParser()

chain_1 = prompt_1 | model | parser
chain_2 = prompt_2 | model | parser
full_chain = chain_1 | chain_2

response_1 = chain_1.invoke({"topic" : "Gas turbine"})
full_response = full_chain.invoke({"topic" : "Gas turbine"})


print(response_1) # This will generate  a detailed report on topic

print("\n ======================================================\n")

print(full_response) # This will print a 3 point summary of the detailed report on gas turbine
