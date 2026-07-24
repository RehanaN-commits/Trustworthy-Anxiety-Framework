from pathlib import Path
import whisper

print("[INFO] Loading Whisper model...")

model = whisper.load_model("base")

print("[INFO] Whisper loaded successfully.")


def transcribe_audio(audio_path: str):
    """
    Convert speech audio to text.

    Parameters
    ----------
    audio_path : str
        Path to audio file.

    Returns
    -------
    str
        Transcript.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"{audio_path} not found.")

    result = model.transcribe(str(audio_path))

    return result["text"].strip()


if __name__ == "__main__":

    transcript = transcribe_audio("sample.wav")

    print("\nTranscript\n")

    print(transcript)