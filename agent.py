from faster_whisper import WhisperModel
import json
import requests

from openai import OpenAI


env = json.loads(open("env.json").read())
client = OpenAI(api_key=env["openai"])
# API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
# API_TOKEN = env["open"]
# headers = {"Authorization": f"Bearer {API_TOKEN}"}


class Agent:
    def __init__(self, lang="en") -> None:
        self.language = lang

    def transcribe(self, audio_file: str) -> str:
        transc = self.whisper(audio_file)
        return transc.text

    def full_pipeline(self, audio_file: str) -> str:
        transc = self.whisper(audio_file)
        answers = self.ask_gpt(transc.text)
        print(answers)
        return answers.choices[0].message.content

    def ask_gpt(self, prompt):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant in a scientific exposition/park for kids, keep your answers short.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response

    def whisper(self, filename):
        audio_file = open(filename, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language=self.language
        )
        print(transcript)
        return transcript
