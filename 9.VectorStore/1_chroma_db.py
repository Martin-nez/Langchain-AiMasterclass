from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

texts = [
    "Large language models are trained on large dataset.",
    "Large language models(llms) are particularly transformers.",
    "Chroma is a lightweight vector store used in langchain",
    "Embeddings convert text into numerical representations"
]

embedding = OpenAIEmbeddings(model = "text-embedding-3-small")

vectorstore = Chroma.from_texts(texts= texts, embedding=embedding, collection_name="langchain_store") # The from_texts method is a convenient way to create a Chroma vector store from a list of texts. You need to provide the texts, the embedding model, and a collection name for the vector store. The collection name is used to organize your data within Chroma.

query = "Tell me about large language models"

results = vectorstore.similarity_search(query=query, k=2) # The similarity_search method is used to perform a similarity search on the vector store. You need to provide the query and the number of results you want to retrieve (k). The method will return the most similar texts from the vector store based on the embeddings.

# print(result)   uncomment this to see the full raw result


for i , result in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: {result.page_content}")


# Make sure you install chromadb and langchain-chroma using
# pip install chromadb langchain-chroma or uv add chromadb langchain-chroma