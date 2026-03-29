from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load + split
loader = TextLoader("data.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# Embeddings + vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vectorstore.as_retriever()

# LLM (simple local HF model)
# pipe = pipeline("text-generation", model="gpt2")
pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=100
)
llm = HuggingFacePipeline(pipeline=pipe)

# Query
query = "What is RAG?"

# Retrieve
# retrieved_docs = retriever.get_relevant_documents(query)
retrieved_docs = retriever.invoke(query)
context = " ".join([doc.page_content for doc in retrieved_docs])

# Final prompt
final_prompt = f"Answer based on context: {context}\nQuestion: {query}"

response = llm.invoke(final_prompt)

# print(response[0]["generated_text"])
print(response)