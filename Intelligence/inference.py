# TODO: Define a function with langchain which gives inference.
# TODO: Define a function with Gamma which can do sentiment analysis
# get user requirements
# return sentimental analysis and recommendation
# TODO:Write a function which gets xlxs, csv files and returns
#  recommendation and sentimental analysis.


import os
from langchain.prompts import ChatPromptTemplate
from ai_engine import groq_api

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
        Sentiment (choose from Positive, Negative, Neutral): 
        """
    )

    # Create a chain with the LLM and the prompt
    chain = prompt | llm

    # Run the chain with the given input
    response = chain.invoke({"text": text})
    return response.content


# Example usage
if __name__ == "__main__":
    text_input = "I am extremely happy with the service provided!"
    sentiment = analyze_sentiment(text_input)
    print(f"Sentiment: {sentiment}")
