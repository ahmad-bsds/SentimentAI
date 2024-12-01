from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Initialize the LLM
llm = OpenAI(model="gpt-4", temperature=0)

# Create a pandas DataFrame agent
agent = create_pandas_dataframe_agent(llm, df, verbose=True)
