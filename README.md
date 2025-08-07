# AI Assistant in a Box

This project provides a containerized, robust, and reproducible environment for a powerful AI assistant. It comes pre-configured with high-quality, local models for all core AI tasks: Text-to-Speech (TTS), Automatic Speech Recognition (ASR), and a Large Language Model (LLM).

## Core Components

The Docker container includes the following core components:

*   **Google's Gemma 3n:** The `google/gemma-3n-E4B-it` model is used for state-of-the-art language understanding and generation.
*   **Coqui TTS**: The `tts_models/en/ljspeech/vits` model is used for high-quality, local text-to-speech synthesis.
*   **Vakyansh ASR**: An Indic-language ASR toolkit is used for local speech recognition. The application automatically downloads the necessary English model and starts the ASR server.

## How to Use

### 1. Build the Docker Image

First, you need to build the Docker image from the `Dockerfile`. All dependencies, including the Vakyansh git repository, are pinned to specific versions to ensure a reproducible build.

Open a terminal in the project root and run:
```bash
docker build -t ai-assistant .
```
This process will take some time as it downloads all the dependencies.

### 2. Run the Docker Container

Once the image is built, you can run the container. The container will start the `assistant.py` script. On the first run, the script will download the necessary ASR models from Vakyansh, which may take a few minutes. Subsequent runs will be faster.

```bash
docker run --rm -it ai-assistant
```

**Note on GPUs:** The included AI libraries are computationally intensive and will run much faster with a GPU. If you have a compatible NVIDIA GPU and the NVIDIA Container Toolkit installed, you can enable GPU access for the container like this:

```bash
docker run --rm -it --gpus all ai-assistant
```

### What the Script Does

The `assistant.py` script is a fully-featured AI assistant that:

1.  **Initializes Services**: It downloads the Vakyansh English ASR model and its dictionary, then starts the Vakyansh gRPC server in the background. It also initializes the Coqui TTS and Gemma 3n models.
2.  **Waits for Server Readiness**: It actively probes the Vakyansh server to ensure it's fully ready before proceeding.
3.  **Interactive Loop**: It enters an interactive loop where you can type commands to the assistant. The assistant can:
    *   Speak text using Coqui TTS.
    *   Listen to your voice and transcribe it using the local Vakyansh ASR server.
    *   Take screenshots.
    *   Run shell commands.
    *   Answer questions using the Gemma 3n language model.
    *   Maintain context using LangChain's combined memory stack—full conversation buffer, windowed buffer, summary, and vector memories.
