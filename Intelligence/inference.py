import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from Intelligence.ai_engine import groq_api
from utils import get_logger
from Intelligence.memory import chat_with_history, prompt_template

# Adding a logger
logger = get_logger(__name__)

# Adding Gemma model from groq api
groq_llm = groq_api()


# Function to perform sentiment analysis
def analyze_sentiment(text, session_id):
    """
    Function to perform sentiment analysis.
    
    :param session_id: 
    :param text: text
    :return: sentiment decision
    """""

    llm = groq_llm

    # Create a chain with the LLM and the prompt
    chain = prompt_template | llm

    chat = chat_with_history(process_chain=chain)

    # Run the chain with the given input
    response_context = chat.invoke(
        {"question": text},
        config={"configurable": {"session_id": session_id}},

    )

    logger.info("Analyzing.....")
    context = response_context.content
    logger.info("Analyzed")

    # Sentiment
    return context


def recommendations(session_id):

    prompt_template_ = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            ("human", "Utilise existing history and answer this: {question}")
        ]
    )

    # Create a chain with the LLM and the prompt
    chain = prompt_template_ | groq_llm

    chat = chat_with_history(process_chain=chain)

    # Run the chain with the given input
    response_context = chat.invoke(
        {"question": "What you have to tell me?"},
        config={"configurable": {"session_id": session_id}},

    )

    return response_context
