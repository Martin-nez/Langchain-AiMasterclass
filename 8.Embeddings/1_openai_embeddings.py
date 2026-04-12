from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# text = "Lagos is a busy city in Nigeria, with a population of over 14 million people. It is known for its vibrant culture, bustling markets, and beautiful beaches."
# result = embedding.embed_query(text)
# print(result)

vector = embedding.embed_query("what is langchain?")
print(f"\nLength of vector: {len(vector)}\n")
print("="*40)
print(vector)
