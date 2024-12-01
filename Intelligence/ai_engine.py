from langchain_groq import ChatGroq
from utils import load_env_variable
import os


def groq_api(model: str = "gemma2-9b-it") -> ChatGroq:
    """
    Initializes the GROQ API client.
    Args:
        model (str): The model to use for the API.
    Returns:
        ChatGroq: An instance of the ChatGroq model.
    """
    try:
        os.environ["GROQ_API_KEY"] = load_env_variable("GROQ_API_KEY", env_file_path="../.env")
        llm = ChatGroq(
            temperature=0,
            groq_api_key=load_env_variable("GROQ_API_KEY", env_file_path="../.env"),
            model_name=model,
        )
        return llm
    except Exception as e:
        print(f"Error initializing GROQ API: {e}")
        raise



