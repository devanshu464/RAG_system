## Data Ingestion
## PDF query
from langchain_community.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("/home/devanshu/documemts/resume.pdf")
pdf_text = pdf_loader.load()
#print(pdf_text)

## Data Transfromation
# splitting text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
texts = text_spliter.split_documents(pdf_text)
#print(texts)

## Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(texts,OllamaEmbeddings())
#print(db)

## LLM Model Init
from langchain_community.llms import Ollama
## load llama3 using ollama
llm = Ollama(model='llama3')
#print(llm)

## Design chatprompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.
        <context>
        {context}
        </context>
        Questions : {input}"""
        )

#print(prompt)

## chain introduction
# create stuff document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm,prompt)
#print(document_chain)

## retriever
retriever = db.as_retriever()

## retrieval chain
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever,document_chain)

## user query
response = retrieval_chain.invoke({"input":"summarize about test case remarks and give some diagonistic recommendation if it is needed. In diagonistic recommendation, do not give client side recommendation"})
#response = retrieval_chain.invoke({"input":"explain about testcase reports only"})
print(response["answer"])





