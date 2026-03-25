from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from document_chunking import ai_chunks, insurance_chunks

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 모든 청크를 하나의 리스트로 합친다
all_chunks = ai_chunks + insurance_chunks

# Chroma DB를 생성하고 영구적으로 저장할 경로를 지정한다
persist_directory = "chroma_data"

# Chroma DB 생성 및 저장
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)
