# AI Assistant in a Box

This project provides a containerized environment for an AI assistant with Text-to-Speech (TTS), Automatic Speech Recognition (ASR), and a Large Language Model (LLM).

## Dependencies

The Docker container includes the following core components:

*   **Coqui TTS**: For text-to-speech synthesis.
*   **Vakyansh ASR**: An Indic-language ASR toolkit. The container is set up to run the Vakyansh gRPC server.
*   **Google's Gemma**: A lightweight, state-of-the-art open language model, accessed via the `transformers` library.
*   **PyTorch**: As the backend for the deep learning models.

## How to Use

### 1. Build the Docker Image

First, you need to build the Docker image from the `Dockerfile`. Open a terminal in the project root and run:

```bash
docker build -t ai-assistant .
```

This process will take some time as it downloads all the dependencies and models.

### 2. Run the Docker Container

Once the image is built, you can run the container. The container will start the `assistant.py` script, which demonstrates the capabilities of the included libraries.

```bash
docker run --rm -it ai-assistant
```

**Note on GPUs:** The included AI libraries are computationally intensive and will run much faster with a GPU. If you have a compatible NVIDIA GPU and the NVIDIA Container Toolkit installed, you can enable GPU access for the container like this:

```bash
docker run --rm -it --gpus all ai-assistant
```

### What the Script Does

The `assistant.py` script performs the following actions:

1.  **Initializes Services**: It downloads a placeholder Vakyansh model and starts the Vakyansh ASR gRPC server in the background.
2.  **Text-to-Speech**: It uses Coqui TTS to convert the sentence "Hello, this is a test of the text to speech system." into an audio file (`output.wav`).
3.  **Speech-to-Text (Placeholder)**: It demonstrates where you would call the Vakyansh ASR service to transcribe the generated audio file. The actual transcription is a placeholder in the script.
4.  **Language Model**: It uses the Gemma 2B model to answer the question: "What is the primary function of a CPU in a computer?".
5.  **Vocalizes Response**: It uses Coqui TTS to convert Gemma's answer into speech (`gemma_response.wav`).
6.  **Cleans Up**: It stops the Vakyansh server before exiting.
