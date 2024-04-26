from fastapi import FastAPI, UploadFile, File, HTTPException,  Header, Depends
from fastapi.responses import JSONResponse
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from document_handling import text_split, load_pdf, load_unstructured_pdf
from summarizer import summary_workflow,summarize
import shutil
from summarizer2 import generate, process_document

OPEN_API_KEY = ''

os.environ["OPENAI_API_KEY"]  = OPEN_API_KEY



app = FastAPI()


emb_model = OpenAIEmbeddings()
workflow = summary_workflow()
retriever = None
text_chunks = None

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    with open("temp.pdf", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    global retriever, summary 
    try:
        # Load PDF data from uploaded file
        #data = load_pdf("temp.pdf") 
        data = load_unstructured_pdf("temp.pdf")
        text_chunks = text_split(data)

        # Embed and store document chunks
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            collection_name="doc-storage",
            embedding=emb_model)

        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.5,"k": 5})
        summary = summarize(text_chunks)
        process_document(data) 
        if retriever:
            return {"message": "Document processed and stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_document(question: str):
    global retriever, summary
    if not retriever :
        raise HTTPException(status_code=400, detail="No document processed. Upload a document first.")

    try:
        inputs = {"keys": {"question": question, "retriever": retriever, "summary": summary}}
        generated_answer = None
        for output in workflow.stream(inputs):
            for key, value in output.items():
                pass
        generated_answer = value['keys']['generation']

        return {"answer": generated_answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/summarize-query")
async def summarize_query(question:str):
    global retriever, summary
    if not retriever :
        raise HTTPException(status_code=400, detail="No document processed. Upload a document first.")
        
    try:
        generated_answer = None

        generated_answer = generate(question)
        return {"answer": generated_answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))