from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 저장된 벡터 저장소를 불러온다(생성이 아님!)
retrieved_vectorstore = Chroma(
    persist_directory="chroma_data",
    embedding_function=embedding_model,
)

# 불러오기가 성공했는지 확인
collection = retrieved_vectorstore.get()
# print(f"총 {len(collection['ids'])}개 문서 확인")

# 필터 조건 설정
ai_filter = {
    'source_type': 'glossary',
}

# 검색기(Retriever) 객체를 생성한다
retriever = retrieved_vectorstore.as_retriever(
    search_kwargs={
        'filter': ai_filter,
    },
)

# 검색어를 사용하여 문서를 검색
query = "파이썬 반복문에 대해 알려줘"
docs = retriever.invoke(query)

print(f"검색 결과 문서 수: {len(docs)}개")
for i in range(len(docs)):
    print(f"\n--- 문서 {i+1} ---")
    print(docs[i].page_content)
