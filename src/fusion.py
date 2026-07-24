"""
====================================================
MULTIMODAL FUSION
====================================================

Combines

1. RoBERTa Prediction
2. Acoustic Features

Acoustic features DO NOT classify the user.

Instead, they provide supporting evidence
for the text prediction.
"""

from typing import Dict


# -----------------------------------------------------
# Acoustic Evidence
# -----------------------------------------------------

def analyze_acoustic_features(features: Dict):

    evidence = []

    score = 0

    # -------------------------------------------------
    # Speech Rate
    # -------------------------------------------------

    speech_rate = features["speech_rate"]

    if speech_rate < 2.0:

        evidence.append("Slow speech")

        score += 1

    elif speech_rate > 4.5:

        evidence.append("Fast speech")

    else:

        evidence.append("Normal speech")

    # -------------------------------------------------
    # Energy
    # -------------------------------------------------

    energy = features["energy"]

    if energy < 0.05:

        evidence.append("Low vocal energy")

        score += 1

    else:

        evidence.append("Normal vocal energy")

    # -------------------------------------------------
    # Pause Ratio
    # -------------------------------------------------

    pause = features["pause_ratio"]

    if pause > 0.30:

        evidence.append("Frequent pauses")

        score += 1

    else:

        evidence.append("Normal pauses")

    # -------------------------------------------------
    # Pitch
    # -------------------------------------------------

    pitch = features["pitch"]

    if pitch < 120:

        evidence.append("Low pitch")

    elif pitch > 280:

        evidence.append("High pitch")

    else:

        evidence.append("Normal pitch")

    # -------------------------------------------------

    return {

        "support_score": score,

        "evidence": evidence

    }


# -----------------------------------------------------
# Fusion
# -----------------------------------------------------

def multimodal_fusion(

    prediction,

    confidence,

    acoustic_features

):

    acoustic = analyze_acoustic_features(

        acoustic_features

    )

    support = acoustic["support_score"]

    fused_confidence = confidence

    # -------------------------------------------------
    # Confidence Adjustment
    # -------------------------------------------------

    if support >= 3:

        fused_confidence += 0.05

    elif support == 2:

        fused_confidence += 0.03

    elif support == 1:

        fused_confidence += 0.01

    fused_confidence = min(

        fused_confidence,

        0.99

    )

    # -------------------------------------------------

    return {

        "prediction": prediction,

        "confidence": round(confidence, 3),

        "fused_confidence": round(

            fused_confidence,

            3

        ),

        "acoustic_support": support,

        "acoustic_evidence": acoustic["evidence"],

        "acoustic_features": acoustic_features

    }


# -----------------------------------------------------
# Testing
# -----------------------------------------------------

if __name__ == "__main__":

    acoustic_features = {

        "pitch": 119.21,

        "energy": 0.0717,

        "duration": 9.01,

        "speech_rate": 3.0,

        "pause_ratio": 0.244

    }

    result = multimodal_fusion(

        prediction="Depressed",

        confidence=0.91,

        acoustic_features=acoustic_features

    )

    from pprint import pprint

    pprint(result)