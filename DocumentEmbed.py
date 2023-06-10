from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader


class Embed:
    def __init__(self, persist_directory=None, embedding_function=None):
        self.doc = TextLoader("india.txt").load()  # load Data as Array

        # Split Documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0)
        self.splitted_doc = self.splitter.split_documents(self.doc)
        # print(self.splitted_doc[:10], "\n")

        # Embedding..

        self.doc_vectordb = Chroma.from_documents(
            documents=self.splitted_doc, embedding=embedding_function, persist_directory=persist_directory)
        # print(self.doc_vectordb)
        self.doc_vectordb.persist()
