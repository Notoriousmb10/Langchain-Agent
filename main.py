
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

loader = TextLoader("football_data/data1.txt", encoding="utf-8")
documents = loader.load()

text_splitted = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitted.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)


query = "Why do user loves football so much?"

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

results = retriever.invoke(query)

context = "\n\n".join(
    result.page_content for result in results
)

class LLMResponse(BaseModel):
    text: str
    

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""You are a helpful assistant, that answers users queries, using the context
        Look for the answer for the user provided query in the context. 
        
        Context: {context}
        
        If you are able to find the answer in the context, provide the answer according to the query. If you are unable to find the answer in the context, respond with "I don't know".
        Please answer the query in a concise and informative manner."""
    ),
    ("human", "{query}")
])

structured_llm = llm.with_structured_output(LLMResponse)

chain = prompt | structured_llm

response = chain.invoke({
    "query": "Whats the name of the football player that the user idolizes the most?",
    "context": context
})

print(response)
