import librosa
import numpy as np
import parselmouth
import whisper


# ---------------------------------------------------------
# Acoustic Feature Extraction
# ---------------------------------------------------------

def extract_acoustic_features(audio_path: str):

    y, sr = librosa.load(audio_path, sr=16000)

    # =====================================================
    # MFCC
    # =====================================================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    # =====================================================
    # Pitch (Praat)
    # =====================================================

    sound = parselmouth.Sound(audio_path)

    pitch = sound.to_pitch()

    pitch_values = pitch.selected_array["frequency"]

    pitch_values = pitch_values[pitch_values > 0]

    if len(pitch_values) > 0:
        mean_pitch = float(np.mean(pitch_values))
    else:
        mean_pitch = 0.0

    # =====================================================
    # Energy
    # =====================================================

    rms = librosa.feature.rms(y=y)

    mean_energy = float(np.mean(rms))

    # =====================================================
    # Duration
    # =====================================================

    duration = float(
        librosa.get_duration(
            y=y,
            sr=sr
        )
    )

    # =====================================================
    # Speech Rate (Whisper)
    # =====================================================

    model = whisper.load_model("base")

    result = model.transcribe(
        audio_path,
        fp16=False
    )

    text = result["text"]

    words = text.split()

    num_words = len(words)

    if duration > 0:
        speech_rate = num_words / duration
    else:
        speech_rate = 0.0

    # =====================================================
    # Pause Detection
    # =====================================================

    intervals = librosa.effects.split(
        y,
        top_db=30
    )

    total_speech = 0

    for start, end in intervals:
        total_speech += (end - start)

    total_speech = total_speech / sr

    total_pause = max(0, duration - total_speech)

    pause_ratio = total_pause / duration if duration > 0 else 0

    # =====================================================
    # Return
    # =====================================================

    return {

        "mfcc": mfcc_mean,

        "pitch": round(mean_pitch, 2),

        "energy": round(mean_energy, 4),

        "duration": round(duration, 2),

        "speech_rate": round(speech_rate, 2),

        "pause_ratio": round(pause_ratio, 3),

        "transcript": text.strip()

    }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    AUDIO = "sample.wav"

    features = extract_acoustic_features(AUDIO)

    print("\nTranscript\n")
    print(features["transcript"])

    print("\nAcoustic Features\n")

    print("Pitch :", features["pitch"], "Hz")

    print("Energy :", features["energy"])

    print("Duration :", features["duration"], "sec")

    print("Speech Rate :", features["speech_rate"], "words/sec")

    print("Pause Ratio :", features["pause_ratio"])

    print("\nMFCC")

    print(features["mfcc"])