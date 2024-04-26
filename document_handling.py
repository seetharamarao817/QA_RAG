from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, PDFMinerLoader



def load_pdf(data):
    loader = PyPDFLoader(data)
    documents = loader.load()
    return documents

#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

def load_unstructured_pdf(path):
    loader = PDFMinerLoader(path)
    docs = loader.load()
    return docs