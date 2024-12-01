import pandas as pd
from inference import analyze_sentiment

# Loading data
data = pd.read_csv("../Data/sentiment_analysis.csv")

# New empty column for recommendation
data["y"] = ""


# TODO: Import sentimental function
for text in data["text"]:
    for y in data["text"]:
        y = text.apply(analyze_sentiment)

data["y"] = data["text"].apply(analyze_sentiment)

# TODO: Analyze sentiment through pandas.
print(
    data["y"]
)

