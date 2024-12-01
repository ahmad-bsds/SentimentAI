import os
from langchain.prompts import ChatPromptTemplate
from Intelligence.ai_engine import groq_api
from utils import get_logger
from Intelligence.memory import chat_with_history

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

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI that performs sentiment analysis. Analyze the following text and provide the sentiment:
        Text: "{question}"
        Output formate:
        (sentiment)
        Instructions:
        - Do not include any text other than the sentiment.
        - Return only the sentiment in the output.
        - Do not hallucinate.
        - Think before decision making, and always provide correct answer.
        - Don't include other character like: \n, \t, etc.
        - Do not make variations in sentiment, make your answer final.
        _ Only add 'Positive', 'Negative','Irrelevant' and 'Neutral' word.
        """
    )

    # Create a chain with the LLM and the prompt
    chain = prompt | llm

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

    prompt = ChatPromptTemplate.from_template("Bsed on the above sentiment, suggest me the important recommendations to"
                                              "enhance my business.")

    # Create a chain with the LLM and the prompt
    chain = prompt | groq_llm

    chat = chat_with_history(process_chain=chain)

    # Run the chain with the given input
    response_context = chat.invoke(
        {"question": "suggest me viable recommendations."},
        config={"configurable": {"session_id": session_id}},

    )

    return response_context
