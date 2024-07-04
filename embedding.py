import os

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from PyPDF2 import PdfReader

def create_index(file_path: str) -> None:

    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    with open('output.txt', 'w') as file:
        file.write(text)

    loader = DirectoryLoader(
        './',
        glob='**/*.txt',
        loader_cls=TextLoader
    )

    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1024,
        chunk_overlap=128
    )

    texts = text_splitter.split_documents(documents)

    embed = OpenAIEmbeddings(openai_api_key="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")

    embed_dir = 'embed_dir'

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embed,
        persist_directory=embed_dir
    )

    vectordb.persist()

create_index('saatva_product_details.pdf')