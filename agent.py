from faster_whisper import WhisperModel
import json
import requests
from pydub import AudioSegment
import av
import time
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
        self.grp = grp

    def transcribe(self, audio_file: str) -> str:
        transc = self.whisper(audio_file)
        return transc.text

    def full_pipeline(self, audio_file: str) -> str:
        start_time = time.time()
        self.convert_to_mp3(audio_file, "input.mp3")
        conv_time = time.time() - start_time
        print(f"Conversion time: {conv_time} seconds")

        start_time = time.time()
        transc = self.whisper("input.mp3")
        whisper_time = time.time() - start_time
        print(f"Whisper time: {whisper_time} seconds")

        start_time = time.time()
        answers = self.ask_gpt(transc.text)
        gpt_time = time.time() - start_time
        print(f"GPT time: {gpt_time} seconds")

        start_time = time.time()
        ans_text = answers.choices[0].message.content
        ans_audio = self.tts(ans_text)
        tts_time = time.time() - start_time
        print(f"TTS time: {tts_time} seconds")

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
        model = "eleven_turbo_v2" if self.language == "en" else "eleven_multilingual_v2"
        actor = env["voices"][self.grp]
        print(actor)
        audio = generate(
            text=text,
            voice=actor,
            api_key=env["elv"],
            model=model,
        )
        save(audio, "output/output.mp3")

        return audio

    def convert_to_mp3(self, webm_file_path, mp3_file_path):
        try:
            with av.open(webm_file_path, "r") as inp:
                # f = SpooledTemporaryFile(mode="w+b")
                f = mp3_file_path
                with av.open(
                    f, "w", format="mp3"
                ) as out:  # Open file, setting format to mp3
                    out_stream = out.add_stream("mp3")
                    for frame in inp.decode(audio=0):
                        frame.pts = None
                        for packets in out_stream.encode(frame):
                            out.mux(packets)
                    for packets in out_stream.encode(None):
                        out.mux(packets)

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
