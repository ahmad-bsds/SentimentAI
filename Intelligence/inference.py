# get user requirements
# return sentimental analysis and recommendation
# TODO:Write a function which gets xlxs, csv files and returns
#  recommendation and sentimental analysis.


import os
from langchain.prompts import ChatPromptTemplate
from ai_engine import groq_api

groq_llm = groq_api()


# Function to perform sentiment analysis
def analyze_sentiment(text, user_requirement):
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
        Text: "{text}" based on this {user_requirement}.\
        Output formate:
        Sentiment: str(sentiment)
        Recommendation: str(recommendation)
        """
    )

    # Create a chain with the LLM and the prompt
    chain = prompt | llm

    # Run the chain with the given input
    response = chain.invoke({
                            "text": text,
                            "user_requirement": user_requirement,
                             })

    return response.content


# Example usage
if __name__ == "__main__":
    text_input = """
                Retrieve user reviews for the product.

    User Reviews:
    1. "This product exceeded my expectations! The build quality is excellent, and the performance is top-notch. Highly recommend it to everyone!" 
       - Sarah J. ⭐⭐⭐⭐⭐

    2. "Good value for money. The features are great, but the battery life could be improved. Overall, I'm satisfied with my purchase." 
       - Michael K. ⭐⭐⭐⭐

    3. "I faced some issues with the setup process, but customer service was quick to help me out. Now it works perfectly!" 
       - Priya L. ⭐⭐⭐⭐

    4. "The product is okay, but I expected more based on the description. It's functional, but there are better alternatives out there." 
       - Daniel R. ⭐⭐⭐

    5. "Terrible experience! The product arrived damaged, and I couldn’t get a replacement. Very disappointed with the service." 
       - Emily S. ⭐
    
    """
    sentiment = analyze_sentiment(text_input, user_requirement="Improve customer engagement strategies")
    print(f"Inference: {sentiment}")
