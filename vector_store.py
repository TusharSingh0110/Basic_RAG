from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load data
loader = TextLoader("data.txt")
documents = loader.load()

# Split
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# Test retrieval
query = "What is RAG?"
results = vectorstore.similarity_search(query, k=2)

for res in results:
    print(res.page_content)