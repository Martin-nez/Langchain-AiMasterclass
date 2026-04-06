from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.7
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that extracts information from web pages."),
        ("human", "Answer the following question {question} from the following text {text}")
    ]
)

url = "https://www.qoani.com"

loader = WebBaseLoader(web_path=url) #n The WebBaseLoader is used to load web pages. You can specify the URL of the web page you want to load.

documents = loader.load() # The load method is used to load the web page and return the content as a list of documents. Each document contains the text content of the web page.

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({
    "question": "What is Qoani?",
    "text": documents[0].page_content
})

print(response)