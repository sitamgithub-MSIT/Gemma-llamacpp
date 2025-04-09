# Importing required libraries
import warnings
warnings.filterwarnings("ignore")

import os
import json
import subprocess
import sys
from typing import List, Tuple
from llama_cpp import Llama
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.messages_formatter import MessagesFormatter, PromptMarkers
from huggingface_hub import hf_hub_download
import gradio as gr
from logger import logging
from exception import CustomExceptionHandling


# Load the Environment Variables from .env file
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Download gguf model files
if not os.path.exists("./models"):
    os.makedirs("./models")

hf_hub_download(
    repo_id="bartowski/google_gemma-3-1b-it-GGUF",
    filename="google_gemma-3-1b-it-Q6_K.gguf",
    local_dir="./models",
)
hf_hub_download(
    repo_id="bartowski/google_gemma-3-1b-it-GGUF",
    filename="google_gemma-3-1b-it-Q5_K_M.gguf",
    local_dir="./models",
)


# Define the prompt markers for Gemma 3
gemma_3_prompt_markers = {
    Roles.system: PromptMarkers("", "\n"),  # System prompt should be included within user message
    Roles.user: PromptMarkers("<start_of_turn>user\n", "<end_of_turn>\n"),
    Roles.assistant: PromptMarkers("<start_of_turn>model\n", "<end_of_turn>\n"),
    Roles.tool: PromptMarkers("", ""),  # If you need tool support
}

# Create the formatter
gemma_3_formatter = MessagesFormatter(
    pre_prompt="",  # No pre-prompt
    prompt_markers=gemma_3_prompt_markers,
    include_sys_prompt_in_first_user_message=True,  # Include system prompt in first user message
    default_stop_sequences=["<end_of_turn>", "<start_of_turn>"],
    strip_prompt=False,  # Don't strip whitespace from the prompt
    bos_token="<bos>",  # Beginning of sequence token for Gemma 3
    eos_token="<eos>",  # End of sequence token for Gemma 3
)


# Set the title and description
title = "Gemma Llama.cpp"
description = """Google released **[Gemma 3](https://blog.google/technology/developers/gemma-3/)**, a family of multimodal models that offers advanced capabilities like large context and multilingual support.
This interactive chat interface allows you to experiment with the [`gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) text model using various prompts and generation parameters.
Users can select different model variants (GGUF format), system prompts, and observe generated responses in real-time. 
Key generation parameters, such as ⁣`temperature`, `max_tokens`, `top_k` and others are exposed below for tuning model behavior.
For a detailed technical walkthrough, please refer to the accompanying **[blog post](https://sitammeur.medium.com/build-your-own-gemma-3-chatbot-with-gradio-and-llama-cpp-46457b22a28e)**."""


llm = None
llm_model = None

def respond(
    message: str,
    history: List[Tuple[str, str]],
    model: str = "google_gemma-3-1b-it-Q5_K_M.gguf",  # Set default model
    system_message: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
):
    """
    Respond to a message using the Gemma3 model via Llama.cpp.

    Args:
        - message (str): The message to respond to.
        - history (List[Tuple[str, str]]): The chat history.
        - model (str): The model to use.
        - system_message (str): The system message to use.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The temperature of the model.
        - top_p (float): The top-p of the model.
        - top_k (int): The top-k of the model.
        - repeat_penalty (float): The repetition penalty of the model.

    Returns:
        str: The response to the message.
    """
    try:
        # Load the global variables
        global llm
        global llm_model

        # Ensure model is not None
        if model is None:
            model = "google_gemma-3-1b-it-Q5_K_M.gguf"

        # Load the model
        if llm is None or llm_model != model:
            # Check if model file exists
            model_path = f"models/{model}"
            if not os.path.exists(model_path):
                yield f"Error: Model file not found at {model_path}. Please check your model path."
                return

            llm = Llama(
                model_path=f"models/{model}",
                flash_attn=False,
                n_gpu_layers=0,
                n_batch=8,
                n_ctx=2048,
                n_threads=8,
                n_threads_batch=8,
            )
            llm_model = model
        provider = LlamaCppPythonProvider(llm)

        # Create the agent
        agent = LlamaCppAgent(
            provider,
            system_prompt=f"{system_message}",
            custom_messages_formatter=gemma_3_formatter,
            debug_output=True,
        )

        # Set the settings like temperature, top-k, top-p, max tokens, etc.
        settings = provider.get_provider_default_settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.max_tokens = max_tokens
        settings.repeat_penalty = repeat_penalty
        settings.stream = True

        messages = BasicChatHistory()

        # Add the chat history
        for msn in history:
            user = {"role": Roles.user, "content": msn[0]}
            assistant = {"role": Roles.assistant, "content": msn[1]}
            messages.add_message(user)
            messages.add_message(assistant)

        # Get the response stream
        stream = agent.get_chat_response(
            message,
            llm_sampling_settings=settings,
            chat_history=messages,
            returns_streaming_generator=True,
            print_output=False,
        )

        # Log the success
        logging.info("Response stream generated successfully")

        # Generate the response
        outputs = ""
        for output in stream:
            outputs += output
            yield outputs

    # Handle exceptions that may occur during the process
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e


# Create a chat interface
demo = gr.ChatInterface(
    respond,
    examples=[["What is the capital of France?"], ["Tell me something about artificial intelligence."], ["What is gravity?"]],
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Dropdown(
            choices=[
                "google_gemma-3-1b-it-Q6_K.gguf",
                "google_gemma-3-1b-it-Q5_K_M.gguf",
            ],
            value="google_gemma-3-1b-it-Q5_K_M.gguf",
            label="Model",
            info="Select the AI model to use for chat",
        ),
        gr.Textbox(
            value="You are a helpful assistant.",
            label="System Prompt",
            info="Define the AI assistant's personality and behavior",
            lines=2,
        ),
        gr.Slider(
            minimum=512,
            maximum=2048,
            value=1024,
            step=1,
            label="Max Tokens",
            info="Maximum length of response (higher = longer replies)",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.7,
            step=0.1,
            label="Temperature",
            info="Creativity level (higher = more creative, lower = more focused)",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p",
            info="Nucleus sampling threshold",
        ),
        gr.Slider(
            minimum=1,
            maximum=100,
            value=40,
            step=1,
            label="Top-k",
            info="Limit vocabulary choices to top K tokens",
        ),
        gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.1,
            step=0.1,
            label="Repetition Penalty",
            info="Penalize repeated words (higher = less repetition)",
        ),
    ],
    theme="Ocean",
    submit_btn="Send",
    stop_btn="Stop",
    title=title,
    description=description,
    chatbot=gr.Chatbot(scale=1, show_copy_button=True, resizable=True),
    flagging_mode="never",
    editable=True,
    cache_examples=False,
)


# Launch the chat interface
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )
