import pandas as pd
from inference import analyze_sentiment

# Loading data
data = pd.read_csv("../Data/sentiment_analysis.csv")

# New empty column for recommendation
data["y"] = ""


# TODO: Import sentimental function
for col in data["sentiment"]:
    print(col)


