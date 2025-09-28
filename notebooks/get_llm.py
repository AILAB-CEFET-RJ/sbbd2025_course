import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# Load environment variables from .env
load_dotenv()

def get_llm(provider: str = "openai"):
    """
    Return a language model instance configured for either OpenAI or Ollama.

    This function centralizes the initialization of chat-based LLMs so that 
    notebooks and applications can switch seamlessly between cloud-based models 
    (OpenAI) and local models (Ollama).

    Parameters
    ----------
    provider : str, optional
        The backend provider to use. Options are:
        - "openai": returns a ChatOpenAI instance (requires OPENAI_API_KEY in .env).
        - "ollama": returns a ChatOllama instance (requires Ollama installed locally).
        Default is "openai".

    Returns
    -------
    langchain.chat_models.base.BaseChatModel
        A chat model instance that can be invoked with messages.

    Examples
    --------
    Initialize an OpenAI model (requires API key):

    >>> llm = get_llm("openai")
    >>> llm.invoke("Hello, how are you?")

    Initialize a local Ollama model (e.g., Gemma2 2B):

    >>> llm = get_llm("ollama")
    >>> llm.invoke("Summarize the benefits of reinforcement learning.")
    """
    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",  # can also be "gpt-4.1" or "gpt-4o"
            temperature=0
        )
    elif provider == "ollama":
        return ChatOllama(
            model="gemma2:2b",   # replace with any local model installed in Ollama
            temperature=0
        )
    else:
        raise ValueError("Unsupported provider. Use 'openai' or 'ollama'.")
