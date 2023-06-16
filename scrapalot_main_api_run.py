import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from deep_translator import GoogleTranslator
from dotenv import load_dotenv, set_key
from fastapi import UploadFile, FastAPI, Depends, Form, File
from pydantic import BaseModel

from scrapalot_main import get_llm_instance
from scripts.app_environment import translate_docs, translate_src, translate_q, chromaDB_manager, translate_a
from scripts.app_qa_builder import process_database_question, process_query

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
    collection_name: str
    question: str
    locale: str


class TranslationBody(BaseModel):
    locale: str


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
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name], check=True)
    if database_name and collection_name:
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name, "--collection", collection_name], check=True)


@app.post("/set-translation")
async def set_translation(body: TranslationBody):
    locale = body.locale
    set_key('.env', 'TRANSLATE_DST_LANG', locale)
    set_key('.env', 'TRANSLATE_QUESTION', 'true')
    set_key('.env', 'TRANSLATE_ANSWER', 'true')
    set_key('.env', 'TRANSLATE_DOCS', 'true')


@app.post("/upload")
async def upload_documents(database_name: str = Form(...), collection_name: str = Form(None), files: List[UploadFile] = File(...)):
    saved_files = []
    source_documents = './source_documents'
    try:
        for file in files:
            if not collection_name:
                file_path = os.path.join(source_documents, database_name, file.filename)
            else:
                file_path = os.path.join(source_documents, database_name, collection_name, file.filename)

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
    locale = body.locale

    try:
        if translate_q:
            question = GoogleTranslator(source=locale, target=translate_src).translate(question)

        seeking_from = database_name + '/' + collection_name if collection_name else database_name
        print(f"\n\033[94mSeeking for answer from: [{seeking_from}]. May take some minutes...\033[0m")
        qa = process_database_question(database_name, llm, collection_name)
        answer, docs = process_query(qa, question, chat_history, chromadb_get_only_relevant_docs=False, translate_answer=False)

        if translate_a:
            answer = GoogleTranslator(source=translate_src, target=locale).translate(answer)

        source_documents = []
        for doc in docs:
            document_page = doc.page_content.replace('\n', ' ')
            if translate_docs:
                document_page = GoogleTranslator(source=translate_src, target=locale).translate(document_page)

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
