import os
import subprocess
import platform
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from transformers import pipeline
import pyautogui
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

# --- Tool Functions ---

def speak_text(text: str) -> str:
    """
    Converts text to speech and plays it.
    :param text: The text to be spoken.
    """
    try:
        print(f"Speaking: {text}")
        tts = gTTS(text=text, lang='en')
        tts_file = "response.mp3"
        tts.save(tts_file)
        playsound(tts_file)
        os.remove(tts_file)
        return "Text spoken successfully."
    except Exception as e:
        return f"Error in speak_text: {e}"

def listen_to_mic() -> str:
    """
    Listens for audio from the microphone and transcribes it to text.
    NOTE: This requires microphone access from the host machine.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"Heard: {text}")
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

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
    # Using a smaller, local model for demonstration purposes.
    # In a real-world scenario, you might use a more powerful model like Gemma.
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={"max_new_tokens": 200},
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
    # This is a basic ReAct prompt.
    prompt_template = """
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
    # We are not using memory in this simplified example, but here's how you would set it up.
    # memory = ConversationBufferWindowMemory(k=5)

    # 6. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

# --- Main Interactive Loop ---

if __name__ == "__main__":
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

    print("\nAssistant shutting down.")
