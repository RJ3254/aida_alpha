# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    wget \
    # For GUI automation and audio
    scrot \
    python3-tk \
    python3-dev \
    portaudio19-dev \
    libx11-dev \
    libxtst-dev \
    && rm -rf /var/lib/apt/lists/*

# Install all Python libraries
RUN pip install --no-cache-dir \
    # Core AI/ML
    TTS \
    transformers \
    torch \
    accelerate \
    # Vakyansh
    grpcio \
    grpcio-tools \
    # New additions from user
    pyautogui \
    SpeechRecognition \
    vosk \
    openai-whisper \
    keyboard \
    pynput \
    pyaudio \
    sounddevice \
    pyttsx3 \
    edge-tts \
    gTTS \
    playsound \
    simpleaudio \
    langchain \
    chromadb \
    faiss-cpu

# Clone the Vakyansh repository
RUN git clone https://github.com/Open-Speech-EkStep/speech-recognition-open-api.git

# Install Vakyansh dependencies
RUN pip install --no-cache-dir -r speech-recognition-open-api/requirements.txt

# Vakyansh models will be downloaded by the application code
# to avoid bloating the image with models for all languages.

# Copy the application code into the container
COPY assistant.py .

# Command to run the application
CMD ["python", "assistant.py"]
