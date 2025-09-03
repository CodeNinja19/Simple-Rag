from dotenv import load_dotenv

# Loading the env variables
load_dotenv(dotenv_path=".env")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print(embeddings)

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

#### INDEXING ####

# Load Documents
loader = RecursiveUrlLoader(
    url="https://dtu.ac.in/Web/About/history.php",
    max_depth=1,  # how deep to follow links
    extractor=lambda x: bs4.BeautifulSoup(x, "html.parser").text,  # clean text
)
docs = loader.load()

# Cleaning the output
for doc in docs:
    doc.page_content = doc.page_content.replace("\n","").replace("\t","")

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embeddings)

retriever = vectorstore.as_retriever()

# Creating the Prompt manaually
template = """Answer all the questions from the following prompts:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

async def call(msg):
    response = await rag_chain.ainvoke(msg)
    # print(response.pretty_repr(html=True))
    return response