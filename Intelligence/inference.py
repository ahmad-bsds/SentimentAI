import os
from langchain.prompts import ChatPromptTemplate
from Intelligence.ai_engine import groq_api
from utils import get_logger

# Adding a logger
logger = get_logger(__name__)

# Adding Gemma model from groq api
groq_llm = groq_api()


# Function to perform sentiment analysis
def analyze_sentiment(text):
    """
    Function to perform sentiment analysis.
    
    :param text: text
    :return: sentiment decision
    """""

    llm = groq_llm

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI that performs sentiment analysis. Analyze the following text and provide the sentiment:
        Text: "{text}"
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

    # Run the chain with the given input
    response = chain.invoke({
                            "text": text

                             })

    logger.info("Analyzing.....")
    context = response.content
    logger.info("Analyzed")

    # Sentiment
    return context

