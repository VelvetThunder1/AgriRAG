import streamlit as st
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from groq import Groq

class PDFChunker:
    def __init__(self, pdf_path, chunk_size=2000, chunk_overlap=200) -> None:
        self.pages = PyPDFLoader(pdf_path).load_and_split()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self) -> List:
        return self.text_splitter.split_documents(self.pages)

# Initialize the document chunker
chunker = PDFChunker("ClimateChangeAdaptationforAgricultureinDevelopingCountries.pdf")
docs = chunker.chunk()

# Load HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Pinecone API key and create index
PINECONE_API_KEY = "02982fda-9246-4f13-9419-9b219a6c85c4"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
index_name = "rag-hackathon"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Function to create a Pinecone index
def create_index(index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

create_index(index_name)
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

# Initialize Groq client
client = Groq(api_key="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")
messages = [{"role": "system", "content": "You are a helpful assistant"}]

# Chatbot function
def chatbot(question):
    r_docs = docsearch.similarity_search(query=question, k=2)
    context = "\n\n".join(doc.page_content for doc in r_docs)
    
    usr_message = f"""Answer the following question based on the context provided. If you
                do not know the answer, say "I do not know." DO NOT SAY "Based on the given context"
                
                ## Question: 
                {question}

                ## Context:
                {context}
                """
    
    messages.append({"role": "user", "content": usr_message})
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    
    llm_response = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": llm_response})
    return llm_response

# Streamlit UI
st.title("Climate Change Adaptation for Agriculture in Developing Countries RAG")

# Input for user question
question = st.text_input("Enter your question:")

# If a question is entered, process it and display the response
if st.button("Ask"):
    if question:
        response = chatbot(question)
        st.write(response)
    else:
        st.write("Please enter a question.")
