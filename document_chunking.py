from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. AI 용어 문서 불러오기(Text File)
ai_loader = TextLoader("data/Python_AI_Glossary_Guide.txt", encoding="utf-8")
ai_docs = ai_loader.load()

# 2. 펫보험 약관 불러오기(PDF File)
insurance_loader = PyPDFLoader("data/meritz_pet_insurance.pdf")
insurance_docs = insurance_loader.load()

# 데이터 출력
# print(f"AI 용어: {len(ai_docs)}개 문서 불러오기")
# print(f"메타데이터: {ai_docs[0].metadata}")
# print(f"내용: {ai_docs[0].page_content[:500]}...")
# print()
# print(f"펫보험 약관: {len(insurance_docs)}개 페이지 불러오기")
# print(f"메타데이터: {insurance_docs[0].metadata}")
# print(f"내용: {insurance_docs[0].page_content[:500]}...")

# Chunking 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

ai_chunks = text_splitter.split_documents(ai_docs)
insurance_chunks = text_splitter.split_documents(insurance_docs)

# Chunking 결과 출력
# print(f"AI 용어 청킹: {len(ai_chunks)}개 청크로 분할")
# print(ai_chunks[0].page_content)
# print(ai_chunks[1].page_content)
# print()
# print(f"펫보험 청킹: {len(insurance_chunks)}개 청크로 분할")
# print(insurance_chunks[0].page_content)
# print(insurance_chunks[1].page_content)

# 1. AI 용어 청크에 메타데이터 추가
for chunk in ai_chunks:
    chunk.metadata["source_type"] = "glossary"
    chunk.metadata["category"] = "AI & Python"

# 2. 펫보험 청크에 메타데이터 추가
for chunk in insurance_chunks:
    chunk.metadata["source_type"] = "insurance"
    chunk.metadata["category"] = "Pet Insurance"

print(ai_chunks[0].metadata)
print(insurance_chunks[0].metadata)