from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
loader = DirectoryLoader(path="docs", glob="**/*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()
print(documents[0].page_content)