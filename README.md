# RailInfo 🚄

> An intelligent rail information system powered by speech recognition, text-to-speech, and NLP for extracting user intent.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Tech-Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

---

## About
RailInfo is a smart application designed to provide **real-time train information** through natural interaction.  
Users can speak queries in their language, and the system processes them to deliver relevant responses.  

This project is **currently under active development**, and new features are being added regularly.

---

## Features
- 🎤 **Speech-to-Text (STT):**  
  Uses [OpenAI Whisper](https://github.com/openai/whisper) for accurate multilingual speech recognition.  

- 🧠 **Intent Extraction:**  
  Hugging Face transformer models are used to understand user intent (e.g., checking schedules, delays, platforms).  

- 🔊 **Text-to-Speech (TTS):**  
  Responses are generated with [gTTS](https://pypi.org/project/gTTS/) for natural-sounding audio feedback.  

- 🌍 **Multilingual Support:**  
  Queries can be processed in different languages.  

- 📊 **Future Features:**  
  - Real-time train tracking  
  - Integration with station maps  
  - Mobile-friendly UI  

---

## Tech-Stack
- **Speech Recognition:** [OpenAI Whisper](https://github.com/openai/whisper)  
- **Text-to-Speech:** [gTTS](https://pypi.org/project/gTTS/)  
- **NLP & Intent Extraction:** Hugging Face Transformer Models  
- **Backend:** Python (FastAPI / Flask)  
- **Frontend:** HTML, CSS, JavaScript (planned)  
- **Other Tools:** Git, GitHub for version control  

