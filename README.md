# Gemma Llama.cpp

Google released **[Gemma 3](https://blog.google/technology/developers/gemma-3/)**, a family of multimodal models that offers advanced capabilities like large context and multilingual support. This interactive chat interface allows you to experiment with the [`gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) text model using various prompts and generation parameters. Users can select different model variants (GGUF format), system prompts, and observe generated responses in real-time. Key generation parameters, such as ⁣`temperature`, `max_tokens`, `top_k` and others are exposed below for tuning model behavior. For a detailed technical walkthrough, please refer to the accompanying **[blog post](https://sitammeur.medium.com/build-your-own-gemma-3-chatbot-with-gradio-and-llama-cpp-46457b22a28e)**.

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
4. Activate the virtual environment: `tutorial-env\Scripts\activate`
5. Install the required dependencies: `pip install -r requirements.txt`
6. Run the Gradio application: `python app.py`

Now, you can open up your local host and see the web application running. For more information, please refer to the Gradio documentation [here](https://www.gradio.app/docs/interface). Also, a live version of the application can be found [here](https://huggingface.co/spaces/sitammeur/Gemma-llamacpp).

Note: You need a Hugging Face access token to run the application. You can get the token by signing up on the Hugging Face website and creating a new token from the settings page. After getting the token, you can set it as an environment variable `HUGGINGFACE_TOKEN` in your system by creating a `.env` file in the project's root directory.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you want to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions about the project, please contact me on my GitHub profile.

Happy coding! 🚀
