import asyncio
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import unquote

import ebooklib
import mammoth
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from dotenv import load_dotenv, set_key
from ebooklib import epub
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from langchain.callbacks import StreamingStdOutCallbackHandler
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, HTMLResponse

from scrapalot_main import get_llm_instance
from scripts.app_environment import translate_docs, translate_src, translate_q, chromaDB_manager, translate_a, model_n_answer_words
from scripts.app_qa_builder import process_database_question, process_query

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

scheme = "http"
hostname = "127.0.0.1"
port = "8080"
server_url = f'{scheme}://{hostname}:{port}'

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
    translate_chunks: bool = True
    locale: str


class TranslationBody(BaseModel):
    locale: str


class SourceDirectoryDatabase(BaseModel):
    name: str
    path: str


class SourceDirectoryFile(BaseModel):
    id: str
    name: str


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
executor = ThreadPoolExecutor(max_workers=5)


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


async def get_files_from_dir(database: str, page: int, items_per_page: int) -> List[SourceDirectoryFile]:
    all_files = []

    for root, dirs, files in os.walk(database):
        for file in sorted(files):  # Added sorting here.
            if not file.startswith('.'):
                all_files.append(SourceDirectoryFile(id=str(uuid.uuid4()), name=file))
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


async def docx_to_html(docx_path):
    loop = asyncio.get_running_loop()
    with open(docx_path, "rb") as docx_file:
        result = await loop.run_in_executor(executor, mammoth.convert_to_html, docx_file)
        html = result.value  # The generated HTML
    return html


async def epub_to_html(epub_path):
    book = epub.read_epub(epub_path)
    html = "<html><body>"
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content().decode("utf-8"), 'html.parser')
        for img in soup.find_all('img'):
            img.decompose()
        html += str(soup)
    html += "</body></html>"
    return html


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


@app.get("/api/database/{database_name}", response_model=List[SourceDirectoryFile])
async def get_database_files(database_name: str, page: int = Query(1, ge=1), items_per_page: int = Query(10, ge=1)):
    base_dir = "./source_documents"
    absolute_base_dir = os.path.abspath(base_dir)
    database_dir = os.path.join(absolute_base_dir, database_name)
    if not os.path.exists(database_dir) or not os.path.isdir(database_dir):
        raise HTTPException(status_code=404, detail="Database not found")

    files = await get_files_from_dir(database_dir, page, items_per_page)
    return files


@app.get("/api/database/{database_name}/collection/{collection_name}", response_model=List[SourceDirectoryFile])
async def get_database_collection_files(database_name: str, collection_name: str, page: int = Query(1, ge=1), items_per_page: int = Query(10, ge=1)):
    base_dir = "./source_documents"
    absolute_base_dir = os.path.abspath(base_dir)
    collection_dir = os.path.join(absolute_base_dir, database_name, collection_name)
    if not os.path.exists(collection_dir) or not os.path.isdir(collection_dir):
        raise HTTPException(status_code=404, detail="Collection not found")
    files = await get_files_from_dir(collection_dir, page, items_per_page)
    return files


@app.get("/api/database/{database_name}/file/{file_name}", response_model=None)
async def get_database_file(database_name: str, file_name: str) -> Union[HTMLResponse, FileResponse]:
    base_dir = "./source_documents"
    absolute_base_dir = os.path.abspath(base_dir)
    database_dir = os.path.join(absolute_base_dir, database_name)
    if not os.path.exists(database_dir) or not os.path.isdir(database_dir):
        raise HTTPException(status_code=404, detail="Database not found")

    absolute_file_path = os.path.join(database_dir, unquote(file_name))
    if not os.path.exists(absolute_file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_extension = os.path.splitext(absolute_file_path)[-1].lower()

    if file_extension == ".docx":
        html = await docx_to_html(absolute_file_path)
        return HTMLResponse(content=html, status_code=200)
    elif file_extension == ".epub":
        html = await epub_to_html(absolute_file_path)
        return HTMLResponse(content=html, status_code=200)
    else:
        return FileResponse(absolute_file_path)


@app.post('/api/query')
async def query_files(body: QueryBody, llm=Depends(get_llm)):
    database_name = body.database_name
    collection_name = body.collection_name
    question = body.question
    locale = body.locale
    translate_chunks = body.translate_chunks

    try:
        if translate_q:
            question = GoogleTranslator(source=locale, target=translate_src).translate(question)

        seeking_from = database_name + '/' + collection_name if collection_name and collection_name != database_name else database_name
        print(f"\n\033[94mSeeking for answer from: [{seeking_from}]. May take some minutes...\033[0m")
        qa = await process_database_question(database_name, llm, collection_name)
        answer, docs = process_query(qa, question, model_n_answer_words, chat_history, chromadb_get_only_relevant_docs=False, translate_answer=False)

        if translate_a:
            answer = GoogleTranslator(source=translate_src, target=locale).translate(answer)

        source_documents = []
        for doc in docs:
            document_page = doc.page_content.replace('\n', ' ')
            if translate_docs == translate_chunks:
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
async def upload_files(request: Request):
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
    path = 'api'
    # cert_path = "cert/cert.pem"
    # key_path = "cert/key.pem"
    print(f"Scrapalot API is now available at {scheme}://{host}:{port}/{path}")
    uvicorn.run(app, host=host, port=int(port))
    # uvicorn.run(app, host=host, port=int(port), ssl_keyfile=key_path, ssl_certfile=cert_path)
