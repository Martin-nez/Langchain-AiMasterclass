from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="tesla.pdf")
documents = loader.load()

# CharacterTextSplitter splits text based on a separator (like newlines)
# chunk_size: maximum characters per chunk
# chunk_overlap: characters that overlap between chunks for context
# separator: the character/string to split on (default is "\n\n")

splitter = CharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    separator="" 
)


chunks = splitter.split_documents(documents) # The split_documents method is used to split the documents into smaller chunks based on the chunk size, chunk overlap, and separator.

print(f"\nTotal number of chunks: {len(chunks)}\n")
print("="*80)


for i, chunk in enumerate(chunks): # I wrap the chucks here with enumerate because i want to print the index on the list
    print(f"\n--- Chunk {i+1} ---")
    print(f"Content: {chunk.page_content}")
    print(f"Length: {len(chunk.page_content)} characters")
    print(f"Metadata: {chunk.metadata}")
