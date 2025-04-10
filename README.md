# Gemma Llama.cpp

Google released **[Gemma 3](https://blog.google/technology/developers/gemma-3/)**, a family of multimodal models that offers advanced capabilities like large context and multilingual support. This interactive chat interface allows you to experiment with the [`gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) text model using various prompts and generation parameters. Users can select different model variants (GGUF format), system prompts, and observe generated responses in real-time. Key generation parameters, such as ⁣`temperature`, `max_tokens`, `top_k`, and others, are exposed below for tuning model behavior. For a detailed technical walkthrough, please refer to the accompanying **[blog post](https://sitammeur.medium.com/build-your-own-gemma-3-chatbot-with-gradio-and-llama-cpp-46457b22a28e)**.

## Project Structure

The project is structured as follows:

- `app.py`: The file containing the main gradio application.
- `logger.py`: The file containing the code for logging the application.
- `exception.py`: The file containing the code for custom exceptions used in the project.
- `requirements.txt`: The file containing the list of dependencies for the project.
- `LICENSE`: The license file for the project.
- `README.md`: The README file that contains information about the project.
- `assets`: The folder containing screenshots for working on the application.
- `.gitignore`: The file containing the list of files and directories to be ignored by Git.

## Tech Stack

- Python (for the programming language)
- Llama.cpp (llama-cpp-python as Python binding for llama.cpp)
- Hugging Face Hub (for the GGUF model)
- Gradio (for the main application)

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/Gemma-llamacpp.git`
2. Change the directory: `cd Gemma-llamacpp`
3. Create a virtual environment: `python -m venv tutorial-env`
4. Activate the virtual environment:
   - For Linux/Mac: `source tutorial-env/bin/activate`
   - For Windows: `tutorial-env\Scripts\activate`
5. Install the required dependencies: `pip install -r requirements.txt`
6. Run the Gradio application: `python app.py`

Note: You need a Hugging Face access token to run the application. You can get the token by signing up on the Hugging Face website and creating a new token from the settings page. After getting the token, you can set it as an environment variable `HUGGINGFACE_TOKEN` in your system by creating a `.env` file in the project's root directory. Replace the values with your API key.

```bash
HUGGINGFACE_TOKEN=your_token_here
```

Now, you can open up your local host and see the web application running. For more information, please refer to the Gradio documentation [here](https://www.gradio.app/docs/interface). Also, a live version of the application can be found [here](https://huggingface.co/spaces/sitammeur/Gemma-llamacpp).

## Deployment

The application is deployed on Hugging Face Spaces, and you can access it [here](https://huggingface.co/spaces/sitammeur/Gemma-llamacpp). You can host a Gradio demo permanently on the internet using [Hugging Face Spaces](https://huggingface.co/spaces).

After creating a free Hugging Face account, you can deploy your Gradio app with two methods:

- **From the terminal:**  
  Open your terminal in the app directory and run:

  ```bash
  gradio deploy
  ```

  The CLI will gather basic metadata and launch your app. To update your Space, simply re-run this command or enable the GitHub Actions option to automatically update the Space on every git push.

- **From your browser:**  
  Drag and drop a folder containing your Gradio demo and all related files directly on Hugging Face Spaces. For a detailed guide, refer to [this guide on hosting on Hugging Face Spaces](https://huggingface.co/blog/gradio-spaces).

## Usage

To use the application, follow these steps:

1. Open the Gradio interface in your web browser.
2. Select the GGUF model variant you want to use from the dropdown menu. The available options are `google_gemma-3-1b-it-Q6_K.gguf` and `google_gemma-3-1b-it-Q5_K_M.gguf`.
3. Enter your prompt in the text box provided.
4. Adjust the generation parameters as needed. The available parameters are:
   - `system_prompt`: The system prompt that sets the context for the model. You can leave it default or rewrite it as per your requirement.
   - `temperature`: Controls the randomness of the output. Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.2) make it more focused and deterministic.
   - `max_tokens`: The maximum number of tokens to generate in the response.
   - `top_k`: The number of highest probability vocabulary tokens to keep for top-k sampling.
   - `top_p`: The cumulative probability of parameter options to keep for nucleus sampling.
   - `repetition_penalty`: The parameter for repetition penalty.
5. Click the "Send" button to generate a response from the model and the "Stop" button to stop the generation process.
6. The generated response will be displayed in the chat interface above the input box as conversation history.
7. You can also Undo, Retry, and Delete the conversation history in the chat interface:
   - "Undo" will remove the last message from the conversation history.
   - "Retry" will regenerate the response to the conversation history's last message.
   - "Delete" will reset the chat history and start a new conversation.
8. You can also share the generated response as a text file by clicking the "Download" button.
9. You can also copy a message from both the user and the AI by clicking the "Copy message" icon next to the message. This will copy the message to your clipboard.
10. You can also edit a user message by clicking the "Edit" icon next to the message. This will open a text box where you can edit the message. After editing, save the changes.

## Results

The chat interface allows users to interact with the Gemma 3 model in real time. You can enter your prompts and adjust the generation parameters to see how the model responds. For results, refer to the `assets/` directory for the output screenshots, which show the chat interface in action.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you want to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions about the project, please contact me on my GitHub profile.

Happy coding! 🚀
