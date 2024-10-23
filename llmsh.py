#!/usr/bin/env python3
import os
import sys
import logging
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import Ollama
from langchain.tools import ShellTool
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Set up ShellTool
logger.debug("Setting up ShellTool")
shell_tool = ShellTool()

# Prompt Template
template = """
You are an expert in using shell commands. The command will be directly executed in a shell. Only provide a single executable line of shell code as the answer to {question}.
 """
logger.debug("Setting up prompt template")
prompt = ChatPromptTemplate.from_template(template)

# Choose LLM based on environment variable
use_ollama = os.getenv("USE_OLLAMA", "True").lower() == "true"
logger.debug(f"USE_OLLAMA environment variable set to: {use_ollama}")

if use_ollama:
    logger.debug("Using Ollama server for Llama 3")
    llm = Ollama(
        model="llama3.1:70b",  # Replace with actual model name if needed
        verbose=True,
    )
else:
    logger.debug("Using OpenAI Chat API")
    llm = ChatOpenAI(temperature=0)

# Update ShellTool description to escape braces
logger.debug("Updating ShellTool description to escape braces")
shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")

# Initialize Agent
logger.debug("Initializing agent with the selected LLM")
agent = initialize_agent(
    [shell_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    prompt=prompt,
)

# Main function
def main():
    if '-d' in sys.argv or '--debug' in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    logger.debug("Entering main function")
    while True:
        command = input("$ ")
        logger.debug(f"User entered command: {command}")
        if command == "exit":
            logger.debug("Exiting program")
            break
        elif command == "help":
            logger.debug("Displaying help message")
            print("llmsh: a simple natural language shell in python.")
        else:
            try:
                logger.debug(f"Running agent with command: {command}")
                output = agent.run(command)
                logger.debug(f"Agent output: {output}")
                print(output)
            except Exception as e:
                logger.error(f"Exception occurred: {e}
Make sure the model is pulled with `ollama pull llama3.1:70b`.")

if __name__ == "__main__":
    logger.debug("Starting main program")
    main()
