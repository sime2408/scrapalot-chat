import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from fastapi import UploadFile, FastAPI, Depends, Form, File
from pydantic import BaseModel

from scrapalot_main import get_llm_instance
from scripts.user_environment import translate_docs, translate_src, translate_dst, translate_q, chromaDB_manager
from scripts.user_processors import process_database_question, process_query

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

app = FastAPI()

load_dotenv()

# Initialize a chat history list
chat_history = []


class ScrapalotErrorResponse(BaseModel):
    status_code: int
    error: str


class UploadFileBody(BaseModel):
    database_name: str
    files: List[UploadFile]


class QueryBody(BaseModel):
    database_name: str
    collection_name: Optional[str]
    question: str


class LLM:
    def __init__(self):
        self.instance = None

    def get_instance(self):
        if not self.instance:
            self.instance = get_llm_instance()
        return self.instance


llm_manager = LLM()


@app.on_event("startup")
async def startup_event():
    llm_manager.get_instance()


def get_llm():
    return llm_manager.get_instance()


def list_of_collections(database_name: str):
    client = chromaDB_manager.get_client(database_name)
    return client.list_collections()


@app.get("/")
async def root():
    return {"ping": "pong!"}


def run_ingest(database_name: str, collection_name: Optional[str] = None):
    if database_name and not collection_name:
        subprocess.run(["python", "scrapalot_ingest.py", "--ingest-dbname", database_name], check=True)
    if database_name and collection_name:
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name, "--collection", collection_name], check=True)


@app.post("/upload")
async def upload_documents(
        database_name: str = Form(...),
        files: List[UploadFile] = File(...),
        collection_name: Optional[str] = None):

    saved_files = []
    source_documents = './source_documents'
    try:
        for file in files:
            file_path = os.path.join(source_documents, database_name, file.filename)
            saved_files.append(file_path)

            with open(file_path, "wb") as f:
                f.write(await file.read())

        run_ingest(database_name, collection_name)

        response = {
            'message': "OK",
            'files': saved_files,
            "database_name": database_name
        }
        return response
    except Exception as e:
        return ScrapalotErrorResponse(status_code=500, error=str(e))


@app.get('/databases')
async def get_database_names_and_collections():
    base_dir = "./db"
    try:
        database_names = \
            sorted([name for name in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, name))])

        database_info = []
        for database_name in database_names:
            collections = list_of_collections(database_name)
            database_info.append({
                'database_name': database_name,
                'collections': collections
            })

        return database_info
    except Exception as e:
        return ScrapalotErrorResponse(status_code=500, error=str(e))


@app.post('/query')
async def query_documents(body: QueryBody, llm=Depends(get_llm)):
    database_name = body.database_name
    collection_name = body.collection_name
    question = body.question
    try:
        if translate_q:
            question = GoogleTranslator(source=translate_dst, target=translate_src).translate(question)

        print(f"\n\n\033[94mSeeking for answer from: [{database_name}]. May take some minutes...\033[0m")
        qa = process_database_question(database_name, llm, collection_name)
        answer, docs = process_query(qa, question, chat_history, chromadb_get_only_relevant_docs=False)

        source_documents = []
        for doc in docs:
            document_page = doc.page_content.replace('\n', ' ')
            if translate_docs:
                document_page = GoogleTranslator(source=translate_src, target=translate_dst).translate(document_page)

            source_documents.append({
                'content': document_page,
                'link': doc.metadata['source']
            })

        response = {
            'answer': answer,
            'source_documents': source_documents
        }
        return response
    except Exception as e:
        return ScrapalotErrorResponse(status_code=500, error=str(e))

# commented out, because we use web UI
if __name__ == "__main__":
    import uvicorn

    host = '0.0.0.0'
    port = 8080
    print(f"Scrapalot API is now available at http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)
