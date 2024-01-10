from langchain.document_loaders import PyPDFLoader,TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from dotenv import load_dotenv
load_dotenv()
import os


template = """
you are a good AI chat agent , you are provided with chathistory with the user 
you also have the domain knowledge provided which may or may not be helpful in answering the users question
if you feel the domain knowledge can help you in answering the users question feel free to use it
or else you can use your own knowledge and conversation history to answer it

Answer the question based on the following context:

Retrieved Documents:
{retrieved_docs}

Conversation History:
{conversation_history}

Question: {question}
"""
prompt_template = PromptTemplate.from_template(template)

def load_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )
    documents = splitter.split_documents(documents)
    return documents

def load_txt(path):
    loader = TextLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )
    documents = splitter.split_documents(documents)
    return documents


def get_embeddings_model(embedding_model):
    if embedding_model == "openai":
        embeddings = OpenAIEmbeddings()
    else:
        # can chose any model listed at https://huggingface.co/sentence-transformers
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    return embeddings


def dump_to_db(docs, dump_path, embedding_model="all-MiniLM-L6-v2", db_type="faiss"):
    embeddings = get_embeddings_model(embedding_model)
    if db_type == "faiss":
        db = FAISS.from_documents(docs, embeddings)
    else:
        raise ValueError(f"{db_type} not valid")
    db.save_local(dump_path)


def load_db(dump_path, embedding_model):
    embeddings = get_embeddings_model(embedding_model)
    db = FAISS.load_local(dump_path, embeddings)
    return db

def get_conversation_history(memory):
    history = []
    for msg in memory.buffer_as_messages:
        key = msg.dict()
        history += [f"""Actor : {key['type']} , content : {key['content']}"""]
    return "\n".join(history)

def similar_context(db,query):
    res = db.similarity_search(query=query, k=5)
    context = "\n\n".join([doc.page_content for doc in res])
    return context


def query_llm(question,obj):
    db = obj.db
    llm = obj.llm
    memory = obj.memory
    relavant_context = similar_context(db=db,query=question)
    prompt = prompt_template.format(
        retrieved_docs=relavant_context,
        conversation_history=get_conversation_history(memory),
        question=question,
    )
    output = llm.invoke(prompt).content
    memory.save_context({"input": question}, {"output": output})
    return output

class QAObject :
    def __init__(self,file_name,embedding_model="all-MiniLM-L6-v2",db_type = "faiss"):
        self.file_name = file_name
        self.db_path = f"./db/faiss_{file_name}.db"
        self.embedding_model = embedding_model
        self.db_type = db_type
        self.llm = None
        self.db = None
        self.memory = None

def initialise(file_name="transformer",type = "pdf") :
    file_path = f"./docs/{file_name}.{type}"
    db_path = f"./db/faiss_{file_name}.db"
    embedding_model = "all-MiniLM-L6-v2"
    db_type = "faiss"
    if type == "pdf" :
        docs = load_pdf(file_path)
    elif type == "txt" :
        docs = load_txt(file_path)
    else :
        raise ValueError(f"type = {type} is not valid choose between .pdf and .txt files only")
    if not os.path.exists(db_path) :
        dump_to_db(docs=docs,dump_path=db_path,embedding_model=embedding_model,db_type=db_type)
    db = load_db(dump_path=db_path, embedding_model=embedding_model)
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferWindowMemory(return_messages=True, k=10)
    obj = QAObject(file_name=file_name)
    obj.llm = llm
    obj.db = db
    obj.memory = memory
    return obj 


