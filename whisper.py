from faster_whisper import WhisperModel


class WhisperTranscriber:
    def __init__(self, size="tiny") -> None:
        self.model_size = size
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_file: str) -> str:
        segments, info = self.model.transcribe(audio_file, beam_size=5)
        print("Language detected:", info.language)
        return "\n".join([seg.text for seg in segments])
