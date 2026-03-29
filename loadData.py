from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Load file
loader = TextLoader("data.txt")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

docs = text_splitter.split_documents(documents)

# Print chunks
for i, doc in enumerate(docs):
    print(f"Chunk {i}: {doc.page_content}")