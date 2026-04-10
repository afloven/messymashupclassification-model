
# Audio Spectrogram Transformer (AST) for Music Genre Classification

This model was developed to identify musical genres in complex, noisy audio environments. It was built as part of a deep learning project focused on the **Messy Mashup** challenge, where the goal was to classify genres even when multiple instrument tracks and environmental noises overlap.

---

## Model Overview

This model is based on the **Audio Spectrogram Transformer (AST)**. Unlike traditional models that look at audio as a simple sequence of sounds, the AST treats an audio spectrogram like a high-resolution image. It uses an **Attention Mechanism** to look at the entire 10-second clip at once, allowing it to pick up on both subtle instrument textures and overall rhythmic patterns.

### Technical Details
* **Architecture:** Audio Spectrogram Transformer (AST)
* **Sampling Rate:** 16,000 Hz
* **Input Length:** 10 Seconds
* **Feature Type:** Log-Mel Spectrogram (128 Mel bands)
* **Performance:** Achieved a **0.85 Macro F1-score** on the evaluation set.

---

## The Training Strategy: Cross-Song Recombination

The main reason this model performs well is a training technique called **Cross-Song Recombination**. 

In most training setups, a model listens to a single song at a time. To make this model more resilient to noise and "messy" audio, we created synthetic mashups during training. We took individual "stems" (the separate recordings for bass, drums, vocals, and other instruments) from different songs within the same genre and mixed them together. 

We also added environmental noise from the ESC-50 dataset. This forced the model to ignore the chaos and focus only on the core spectral patterns that define a genre—such as the specific frequency of a blues guitar or the tempo of a techno beat.

---

## Supported Genres

The model is trained to classify audio into one of the following 10 categories:
1. **Blues**
2. **Classical**
3. **Country**
4. **Disco**
5. **Hiphop**
6. **Jazz**
7. **Metal**
8. **Pop**
9. **Reggae**
10. **Rock**

---

## How to Use

You can use this model directly with the Hugging Face `transformers` library.

```python
from transformers import pipeline

# Load the model and preprocessor
classifier = pipeline("audio-classification", model="afloven/messymashupclassifier")

# Predict the genre of an audio file
# Make sure your audio is sampled at 16,000 Hz
results = classifier("path_to_your_audio_file.wav")

print(results)
