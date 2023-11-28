from faster_whisper import WhisperModel
import json
import requests
from pydub import AudioSegment
import io
from elevenlabs import generate, play, save

from openai import OpenAI


env = json.loads(open("env.json").read())
client = OpenAI(api_key=env["openai"])
# API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
# API_TOKEN = env["open"]
# headers = {"Authorization": f"Bearer {API_TOKEN}"}


class Agent:
    def __init__(self, lang="en", grp="default") -> None:
        self.language = lang
        self.grp = "mid"

    def transcribe(self, audio_file: str) -> str:
        transc = self.whisper(audio_file)
        return transc.text

    def full_pipeline(self, audio_file: str) -> str:
        transc = self.whisper(audio_file)
        answers = self.ask_gpt(transc.text)
        ans_text = answers.choices[0].message.content
        ans_audio = self.tts(ans_text)
        print(ans_audio)
        # print(answers)
        return ans_audio

    def ask_gpt(self, prompt):
        prompt_sys = (
            env["prompts"]["en"] if self.language == "en" else env["prompts"]["sv"]
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": prompt_sys,
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response

    def ask_gpt_sv(self, prompt):
        pass

    def whisper(self, filename):
        audio_file = open(filename, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language=self.language
        )
        print(transcript)
        return transcript

    # TTS for elv
    def tts(self, text):
        actor = env["voices"][self.grp]
        audio = generate(
            text=text,
            voice=actor,
            api_key=env["elv"],
            model="eleven_multilingual_v2",
        )
        save(audio, "output/output.mp3")

        return audio

    def convert_webm_to_mp3(webm_file_path, mp3_file_path):
        try:
            print(f"Conversion successful: {mp3_file_path}")

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
