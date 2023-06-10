from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from typing import List
from langchain.schema import Document
import os
from DocumentEmbed import Embed

os.environ['OPENAI_API_KEY'] = "ApI-KeY-HeRe"


if __name__ == "__main__":
    embed = OpenAIEmbeddings()
    persist_directory = "persist"

    if not os.path.exists("./"+"persist"):
        embeddedDoc = Embed(embedding_function=embed,
                            persist_directory=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embed)

    model = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())
    while (True):
        q = input("Want to quit type 'quit' else ask question : ")
        if q.strip().lower() == "quit":
            break
        print(model.run(q))
