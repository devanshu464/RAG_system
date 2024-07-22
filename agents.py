## creating tool wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)
#print(wiki_tool.name)

## creating webbaseloader
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)

## creating embeddings
vectordb = FAISS.from_documents(documents,OllamaEmbeddings())

## creating retriever
retriever=vectordb.as_retriever()

## creating langsmith search tool 
from langchain.tools.retriever import create_retriever_tool
langsmith_tool=create_retriever_tool(retriever,'langsmith_search',
        "search for information about langsmith. For any question about langsmith, you have to use this tool")
#print(langsmith_tool.name)

## creating Arxiv tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxvi_tool=ArxivQueryRun(api_wrapper=api_wrapper)
#print(arxvi_tool.name)

## combine tools
tools=[wiki_tool,langsmith_tool,arxvi_tool]
#print(tools)

## llm
# for this code you have to use OPENAI model
from langchain_community.llms import Ollama
llm = Ollama(model='llama3')

## prompt
from langchain import hub
prompt=hub.pull("hwchase17/openai-functions-agent")
#print(prompt.messages)


## agents
from langchain.agents import create_openai_tools_agent
agent=create_openai_tools_agent(llm,tools,prompt)

## agents executor
from langchain.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
#print(agent_executor)

agent_executor.invoke({"input":"tell me about langsmith"})
#response = agent.invoke({"input":"tell me about langsmith"})
#print(response["answer"])




