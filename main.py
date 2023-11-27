from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import Response
from whisper import WhisperTranscriber


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


@app.post("/answer")
def generate_answer(file: UploadFile):
    try:
        # Save the uploaded audio file locally
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as audio_file:
            audio_file.write(file.file.read())
        transcriber = WhisperTranscriber()
        # Transcribe the audio using faster-whisper
        transcription = transcriber.transcribe(file_path)

        # Return the transcription as JSON
        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        # Handle exceptions
        return HTTPException(
            status_code=500, detail=f"Error processing audio: {str(e)}"
        )


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
