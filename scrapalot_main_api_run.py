import os
import subprocess
import sys
import urllib
import uuid
from pathlib import Path
from typing import List, Optional

from deep_translator import GoogleTranslator
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from langchain.callbacks import StreamingStdOutCallbackHandler
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from scrapalot_main import get_llm_instance
from scripts.app_environment import translate_docs, translate_src, translate_q, chromaDB_manager, translate_a, model_n_answer_words
from scripts.app_qa_builder import process_database_question, process_query

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

app = FastAPI(title="scrapalot-chat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

allow_origins = ["http://localhost:3000", "https://scrapalot.com"]

load_dotenv()


###############################################################################
# model classes
###############################################################################
class QueryBody(BaseModel):
    database_name: str
    collection_name: str
    question: str
    locale: str


class TranslationBody(BaseModel):
    locale: str


class SourceDirectoryDatabase(BaseModel):
    name: str
    path: str


class SourceDirectoryFile(BaseModel):
    id: str
    name: str
    path: str


class LLM:
    def __init__(self):
        self.instance = None

    def get_instance(self):
        if not self.instance:
            self.instance = get_llm_instance(StreamingStdOutCallbackHandler())
        return self.instance


###############################################################################
# init
###############################################################################
chat_history = []
llm_manager = LLM()


@app.on_event("startup")
async def startup_event():
    llm_manager.get_instance()


###############################################################################
# helper functions
###############################################################################

def get_llm():
    return llm_manager.get_instance()


def list_of_collections(database_name: str):
    client = chromaDB_manager.get_client(database_name)
    return client.list_collections()


def get_files_from_dir(database: str, page: int, items_per_page: int) -> List[SourceDirectoryFile]:
    all_files = []
    for root, dirs, files in os.walk(database):
        for file in sorted(files):  # Added sorting here.
            if not file.startswith('.'):
                filepath = os.path.join(root, file)
                url_path = f"file://{urllib.parse.quote(filepath)}"
                all_files.append(SourceDirectoryFile(id=str(uuid.uuid4()), name=file, path=url_path))
    start = (page - 1) * items_per_page
    end = start + items_per_page
    return all_files[start:end]


def run_ingest(database_name: str, collection_name: Optional[str] = None):
    if database_name and not collection_name:
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name], check=True)
    if database_name and collection_name:
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name, "--collection", collection_name], check=True)


###############################################################################
# API
###############################################################################
@app.get("/api")
async def root():
    return {"ping": "pong!"}


@app.post("/api/set-translation")
async def set_translation(body: TranslationBody):
    locale = body.locale
    set_key('.env', 'TRANSLATE_DST_LANG', locale)
    set_key('.env', 'TRANSLATE_QUESTION', 'true')
    set_key('.env', 'TRANSLATE_ANSWER', 'true')
    set_key('.env', 'TRANSLATE_DOCS', 'true')


@app.get('/api/databases')
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
        return HTTPException(status_code=500, detail=str(e))


@app.get("/api/databases", response_model=List[SourceDirectoryDatabase])
async def get_database_documents():
    base_dir = "./source_documents"
    directories = []
    for directory in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, directory)
        if os.path.isdir(dir_path):
            directories.append(SourceDirectoryDatabase(id=str(uuid.uuid4()), name=directory, path=dir_path))
    return directories


@app.get("/api/database/{database_name}", response_model=List[SourceDirectoryFile])
async def read_files(database_name: str, page: int = Query(1, ge=1), items_per_page: int = Query(10, ge=1)):
    base_dir = "./source_documents"
    absolute_base_dir = os.path.abspath(base_dir)
    database_dir = os.path.join(absolute_base_dir, database_name)
    if not os.path.exists(database_dir) or not os.path.isdir(database_dir):
        raise HTTPException(status_code=404, detail="Database not found")

    files = get_files_from_dir(database_dir, page, items_per_page)
    return files


@app.get("/api/database/{database_name}/collection/{collection_name}", response_model=List[SourceDirectoryFile])
async def get_collection_files(database_name: str, collection_name: str, page: int = Query(1, ge=1), items_per_page: int = Query(10, ge=1)):
    base_dir = "./source_documents"
    absolute_base_dir = os.path.abspath(base_dir)
    collection_dir = os.path.join(absolute_base_dir, database_name, collection_name)
    if not os.path.exists(collection_dir) or not os.path.isdir(collection_dir):
        raise HTTPException(status_code=404, detail="Collection not found")
    files = get_files_from_dir(collection_dir, page, items_per_page)
    return files


@app.post('/api/query')
async def query_documents(body: QueryBody, llm=Depends(get_llm)):
    database_name = body.database_name
    collection_name = body.collection_name
    question = body.question
    locale = body.locale

    try:
        if translate_q:
            question = GoogleTranslator(source=locale, target=translate_src).translate(question)

        seeking_from = database_name + '/' + collection_name if collection_name and collection_name != database_name else database_name
        print(f"\n\033[94mSeeking for answer from: [{seeking_from}]. May take some minutes...\033[0m")
        qa = process_database_question(database_name, llm, collection_name)
        answer, docs = process_query(qa, question, model_n_answer_words, chat_history, chromadb_get_only_relevant_docs=False, translate_answer=False)

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
        return HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_documents(request: Request):
    form = await request.form()
    database_name = form['database_name']
    collection_name = form.get('collection_name')  # Optional field

    files = form["files"]  # get files from form data

    # make sure files is a list
    if not isinstance(files, list):
        files = [files]

    saved_files = []
    source_documents = './source_documents'
    try:
        for file in files:
            content = await file.read()  # read file content
            if collection_name and database_name != collection_name:
                file_path = os.path.join(source_documents, database_name, collection_name, file.filename)
            else:
                file_path = os.path.join(source_documents, database_name, file.filename)

            saved_files.append(file_path)
            with open(file_path, "wb") as f:
                f.write(content)

            # assuming run_ingest is defined elsewhere
            run_ingest(database_name, collection_name)

            response = {
                'message': "OK",
                'files': saved_files,
                "database_name": database_name
            }
            return response
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


# commented out, because we use web UI
if __name__ == "__main__":
    import uvicorn

    host = '0.0.0.0'
    port = 8080
    path = 'api'
    print(f"Scrapalot API is now available at http://{host}:{port}/{path}")
    uvicorn.run(app, host=host, port=port)
