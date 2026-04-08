from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = "sales_data.csv", encoding= "utf-8")

documents = loader.load()

print(documents[0]) # This will print the first row of the csv file as a document. Each document document contains the text content of the row.
