from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="docs/tesla.pdf") # The PyPDFLoader is used to load PDF files. You can specify the file path. The loader will read the contents of the PDF and return it as a list of documents, where each document corresponds to a page in the PDF.

documents = loader.load()

print(documents[0].page_content)

# Make sure you install pypdf before running this code:
# pip install pypdf or uv add pypdf