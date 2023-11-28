from typing import Union, Annotated
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import Response
from fastapi.responses import FileResponse, StreamingResponse

# from models import Question
from agent import Agent


app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) to allow requests from any origin
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
def transcribe(
    file: Annotated[UploadFile, File()],
    lang: Annotated[str, Form()],
):
    try:
        # file = q.file
        # Save the uploaded audio file locally
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as audio_file:
            audio_file.write(file.file.read())
        transcriber = Agent(lang=lang)
        # Transcribe the audio using faster-whisper
        transcription = transcriber.transcribe(file_path)

        # Return the transcription as JSON
        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        # Handle exceptions
        return HTTPException(
            status_code=500, detail=f"Error processing audio: {str(e)}"
        )


@app.post("/answer")
def gen_answer(
    file: Annotated[UploadFile, File()],
    lang: Annotated[str, Form()],
):
    try:
        # Save the uploaded audio file locally
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as audio_file:
            audio_file.write(file.file.read())
        agent = Agent(lang=lang)
        # Transcribe the audio using faster-whisper
        answer = agent.full_pipeline(file_path)

        # Return the transcription as JSON
        return FileResponse(
            "output/output.mp3",
            media_type="audio/mp3",
        )

    except Exception as e:
        # Handle exceptions
        return HTTPException(
            status_code=500, detail=f"Error processing audio: {str(e)}"
        )
