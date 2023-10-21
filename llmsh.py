#!/usr/bin/env python3
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import ShellTool
from dotenv import load_dotenv
from prompt_toolkit import prompt
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.llms import HuggingFaceHub
# from langchain.llms import HuggingFacePipeline
# Make sure the model path is correct for your system!
import os

load_dotenv()

shell_tool = ShellTool()

# llm = ChatOpenAI(temperature=0)
# model = 'meta-llama/Llama-2-7b-hf'
# llm = HuggingFacePipeline.from_model_id(
#    model_id=model,
#    task="text-generation",
#    model_kwargs={"temperature": 0},
# )


shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")

template = """
You are an expert in using shell commands. Only provide a single executable line of shell code as the answer to {question} 
Question: {question}
The command will be directly executed in a shell. 
Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


self_ask_with_search = initialize_agent(
    [shell_tool],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)


def main():
    while True:
        command = prompt("?")  # input("$ ")
        if command == "exit":
            break
        elif command == "help":
            print("llmsh: a simple natural language shell in python.")
        else:
            self_ask_with_search.run(command)
            # output = chain.invoke({"input": })
            # print(output)


if __name__ == "__main__":
    main()
