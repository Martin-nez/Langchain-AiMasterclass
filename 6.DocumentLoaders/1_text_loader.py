from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path="tesla.txt", encoding="utf-8")

documents = loader.load()

print(documents[0].page_content)