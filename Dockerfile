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

# Install all Python libraries with pinned versions for reproducibility
RUN pip install --no-cache-dir \
    # Core AI/ML
    TTS==0.22.0 \
    transformers==4.38.0 \
    torch==1.7.1 \
    torchaudio==0.7.2 \
    accelerate==0.27.0 \
    # Vakyansh
    grpcio==1.37.0 \
    grpcio-tools==1.37.0 \
    # Other assistant tools
    pyautogui==0.9.54 \
    SpeechRecognition==3.10.0 \
    vosk==0.3.45 \
    openai-whisper==20231117 \
    keyboard==0.13.5 \
    pynput==1.7.6 \
    pyaudio==0.2.14 \
    sounddevice==0.4.6 \
    pyttsx3==2.90 \
    edge-tts==6.1.8 \
    gTTS==2.5.1 \
    playsound==1.3.0 \
    simpleaudio==1.0.4 \
    # LangChain
    langchain==0.1.10 \
    chromadb==0.4.22 \
    faiss-cpu==1.7.4 \
    # Vakyansh dependencies not in the main list
    numpy==1.20.0 \
    GPUtil==1.4.0 \
    pydub==0.25.1 \
    webrtcvad==2.0.11

# Clone the Vakyansh repository
RUN git clone https://github.com/Open-Speech-EkStep/speech-recognition-open-api.git

# Install Vakyansh dependencies
RUN pip install --no-cache-dir -r speech-recognition-open-api/requirements.txt

# Vakyansh models will be downloaded by the application code
# to avoid bloating the image with models for all languages.

# Expose the gRPC port for the Vakyansh ASR server
EXPOSE 50051

# Copy the application code into the container
COPY assistant.py .

# Command to run the application
CMD ["python", "assistant.py"]
