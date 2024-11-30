# TODO: Define a function with langchain which gives inference.
# TODO: Define a function with Gamma which can do sentiment analysis
# get user requirements
# return sentimental analysis and recommendation
# TODO:Write a function which gets xlxs, csv files and returns
#  recommendation and sentimental analysis.
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Function to perform sentiment analysis
def analyze_sentiment(text):
    # Initialize the ChatOpenAI model

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                     openai_api_key="sk-proj-3mNWK2IWefjWxc_JqGYDtDMf78u6dLTyOOshehwu"
                                    "SUxaK2RLcAsVXi3OLo0T5keRRexACuVBM9T3BlbkF"
                                    "JY3DOeu1ldW_6UaZ2yWBteveM8H4i41f1eDSxHnptpqbUZOV"
                                    "-qc8cdLWSz9n-zQoY4sFPNReSgA ")

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI that performs sentiment analysis. Analyze the following text and provide the sentiment:
        Text: "{text}"
        Sentiment (choose from Positive, Negative, Neutral): 
        """
    )

    # Create a chain with the LLM and the prompt
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain with the given input
    response = chain.run({"text": text})
    return response.strip()

# Example usage
if __name__ == "__main__":
    text_input = "I am extremely happy with the service provided!"
    sentiment = analyze_sentiment(text_input)
    print(f"Sentiment: {sentiment}")


