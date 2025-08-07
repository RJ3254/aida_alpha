import os
import subprocess
import platform
import torch
import time
import sys
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory,
)
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import pyautogui
import speech_recognition as sr
import json
import requests
from TTS.api import TTS
from playsound import playsound
import grpc
from speech_recognition_open_api_pb2_grpc import SpeechRecognizerStub
from speech_recognition_open_api_pb2 import (
    Language,
    RecognitionAudio,
    RecognitionConfig,
    SpeechRecognitionRequest,
)


# --- Vakyansh ASR Setup ---

def _download_file(url, path, expected_size):
    """Downloads a file with error handling and size verification."""
    try:
        print(f"Downloading {os.path.basename(path)}...")
        response = requests.get(url, timeout=60, allow_redirects=True, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        downloaded_size = os.path.getsize(path)
        if downloaded_size != expected_size:
            print(f"Error: Downloaded file size ({downloaded_size}) does not match expected size ({expected_size}).")
            os.remove(path) # Clean up corrupted file
            return False

        print(f"Successfully downloaded and verified {os.path.basename(path)}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

def setup_vakyansh():
    """
    Downloads Vakyansh models and creates the necessary configuration files
    if they don't already exist.
    """
    print("Setting up Vakyansh ASR...")
    model_dir = "vakyansh_models"
    english_model_dir = os.path.join(model_dir, "english")
    model_dict_path = os.path.join(model_dir, "model_dict.json")

    # URLs for the English model files
    model_url = "https://storage.googleapis.com/vakyansh-open-models/models/english/en-IN/english_infer.pt"
    dict_url = "https://storage.googleapis.com/vakyansh-open-models/models/english/en-IN/dict.ltr.txt"

    model_path = os.path.join(english_model_dir, "english_infer.pt")
    dict_path = os.path.join(english_model_dir, "dict.ltr.txt")

    # Create directories if they don't exist
    os.makedirs(english_model_dir, exist_ok=True)

    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        if not _download_file(model_url, model_path, expected_size=396627253):
            raise RuntimeError("Failed to download Vakyansh model. Cannot continue.")

    # Download dict if it doesn't exist
    if not os.path.exists(dict_path):
        if not _download_file(dict_url, dict_path, expected_size=241):
            raise RuntimeError("Failed to download Vakyansh dictionary. Cannot continue.")

    # Create model_dict.json if it doesn't exist
    if not os.path.exists(model_dict_path):
        print("Creating model_dict.json...")
        # Note: The path inside the JSON must be the absolute path within the container
        model_config = {
            "en": {
                "path": os.path.abspath(model_path),
                "enablePunctuation": True,
                "enableITN": True
            }
        }
        with open(model_dict_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        print("model_dict.json created.")

    print("Vakyansh ASR setup complete.")
    return model_dir


# --- Initialize TTS ---
# This will download the model on the first run
print("Initializing Coqui TTS...")
tts = TTS("tts_models/en/ljspeech/vits")
print("Coqui TTS initialized.")

# --- Tool Functions ---

def speak_text(text: str) -> str:
    """
    Converts text to speech using Coqui TTS and plays it.
    :param text: The text to be spoken.
    """
    try:
        print(f"Speaking: {text}")
        output_file = "response.wav"
        # Generate speech and save to a file
        tts.tts_to_file(text=text, file_path=output_file)
        # Play the generated audio file
        playsound(output_file)
        # Clean up the audio file
        os.remove(output_file)
        return "Text spoken successfully."
    except Exception as e:
        return f"Error in speak_text: {e}"

def listen_to_mic() -> str:
    """
    Listens for audio from the microphone and transcribes it to text using Vakyansh.
    NOTE: This requires microphone access from the host machine.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio_data = r.listen(source)
        print("Processing...")

    try:
        # Get audio bytes
        audio_bytes = audio_data.get_wav_data()

        # Connect to Vakyansh gRPC server
        host = "localhost"
        port = 50051
        with grpc.insecure_channel(f'{host}:{port}') as channel:
            stub = SpeechRecognizerStub(channel)

            # Prepare the request
            lang = Language(sourceLanguage='en')
            config = RecognitionConfig(
                language=lang,
                audioFormat='wav',
                transcriptionFormat=RecognitionConfig.TranscriptionFormat(value='transcript'),
                punctuation=True,
                enableInverseTextNormalization=True
            )
            audio = RecognitionAudio(audioContent=audio_bytes)
            request = SpeechRecognitionRequest(audio=[audio], config=config)

            # Make the gRPC call
            response = stub.recognize(request)

            if response.status == 'SUCCESS' and response.output:
                transcript = response.output[0].source
                print(f"Heard: {transcript}")
                return transcript
            else:
                error_message = response.status_text or "Unknown error from Vakyansh"
                return f"Vakyansh ASR Error: {error_message}"

    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results from SpeechRecognition service; {e}"
    except grpc.RpcError as e:
        return f"Could not connect to Vakyansh ASR server: {e.details()}"
    except Exception as e:
        return f"An error occurred during speech recognition: {e}"

def take_screenshot(filename: str = "screenshot.png") -> str:
    """
    Takes a screenshot of the entire screen.
    NOTE: This requires the Docker container to have access to the host's display.
    :param filename: The filename to save the screenshot as.
    """
    try:
        pyautogui.screenshot(filename)
        return f"Screenshot saved as {filename}"
    except Exception as e:
        return f"Error taking screenshot: {e}. Ensure container has display access."

def run_os_command(command: str) -> str:
    """
    Runs a command in the operating system's shell.
    :param command: The command to execute.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        return f"Command executed. Output:\n{output}"
    except Exception as e:
        return f"Error running command: {e}"

# --- LangChain Agent Setup ---

def setup_agent():
    """Sets up the LangChain agent with tools, memory, and an LLM."""
    # 1. Initialize the LLM
    # Using the specified Gemma 3n model.
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-3n-E4B-it",
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={"max_new_tokens": 512},
        # Add torch_dtype to load in a more memory-efficient way
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    # 2. Define the tools
    tools = [
        Tool(
            name="SpeakText",
            func=speak_text,
            description="Use this tool to speak a given text aloud. Useful for providing audio responses.",
        ),
        Tool(
            name="ListenToMic",
            func=listen_to_mic,
            description="Use this tool to listen to the user's voice and get a transcription. Use it when the user wants to speak their command.",
        ),
        Tool(
            name="TakeScreenshot",
            func=take_screenshot,
            description="Use this tool to take a screenshot of the screen. It can be saved with an optional filename.",
        ),
        Tool(
            name="RunOSCommand",
            func=run_os_command,
            description="Use this tool to execute a command in the operating system shell. Useful for file operations, system checks, etc.",
        ),
    ]

    # 3. Create the prompt template
    # This is a basic ReAct prompt augmented with conversation history.
    prompt_template = """
    The conversation so far:\n{history}\n
    Summary of prior context:\n{summary}\n
    Relevant facts:\n{vector_memory}\n
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # 4. Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. Set up memory
    buffer_memory = ConversationBufferWindowMemory(k=5, memory_key="history", return_messages=True)
    summary_memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=2000, memory_key="summary"
    )
    full_memory = ConversationBufferMemory(memory_key="full_history", return_messages=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts([], embeddings)
    vector_memory = VectorStoreRetrieverMemory(
        retriever=vector_store.as_retriever(), memory_key="vector_memory"
    )

    memory = CombinedMemory(
        memories=[full_memory, buffer_memory, summary_memory, vector_memory]
    )

    # 6. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

    return agent_executor

# --- Main Interactive Loop ---

if __name__ == "__main__":
    vakyansh_server_process = None
    try:
        # --- Setup and Start Vakyansh Server ---
        vakyansh_model_dir = setup_vakyansh()

        print("Starting Vakyansh ASR server...")
        vakyansh_env = os.environ.copy()
        vakyansh_env["models_base_path"] = os.path.abspath(vakyansh_model_dir)

        # Validate Vakyansh server paths before starting
        stub_path = "speech-recognition-open-api/stub"
        server_path = "speech-recognition-open-api/server.py"

        if not os.path.isdir(stub_path):
            raise FileNotFoundError(f"Vakyansh stub directory not found at: {stub_path}. Was the git clone successful?")
        if not os.path.isfile(server_path):
            raise FileNotFoundError(f"Vakyansh server script not found at: {server_path}. Was the git clone successful?")

        # Add the stub path to sys.path so grpc can find the generated files
        sys.path.append(stub_path)

        vakyansh_server_process = subprocess.Popen(
            ["python", server_path],
            env=vakyansh_env
        )

        # Readiness probe for Vakyansh server
        print("Vakyansh ASR server started. Waiting for it to become ready...")
        max_retries = 20
        for i in range(max_retries):
            try:
                with grpc.insecure_channel('localhost:50051') as channel:
                    grpc.channel_ready_future(channel).result(timeout=1)
                print("Vakyansh ASR server is ready.")
                break
            except grpc.FutureTimeoutError:
                if i < max_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError("Vakyansh ASR server did not become ready in time.")

        # --- Setup LangChain Agent ---
        print("Setting up the AI assistant...")
        ai_assistant = setup_agent()
        print("\nAI Assistant is ready. Type 'exit' to quit.")
        print("-" * 50)

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    break

                response = ai_assistant.invoke({"input": user_input})
                # The actual response is in the 'output' key.
                # We will just print the whole dict for this example.
                print("AI:", response)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An error occurred: {e}")

    finally:
        if vakyansh_server_process:
            print("Terminating Vakyansh ASR server...")
            vakyansh_server_process.terminate()
            vakyansh_server_process.wait()
        print("\nAssistant shutting down.")
