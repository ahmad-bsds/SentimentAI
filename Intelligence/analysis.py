import pandas as pd
from Intelligence.inference import analyze_sentiment, recommendations
from Intelligence.memory import store

session_id = "0000000000000"


def perform_analysis(file_type: str, path: str = "sentiment_analysis.csv", preferences: str = ""):
    if file_type == "csv":
        # Loading CSV data
        data = pd.read_csv(f"../Data/{path}")
    else:
        return "Error! file type not supported."

    # New empty column for recommendation
    data["y"] = ""

    # Sentimental analysis.
    data["y"] = data["text"].apply(lambda text: analyze_sentiment(session_id=session_id, text=text))
    rec = recommendations(session_id=session_id)

    # Results
    df = data.groupby("y")["Month"].count().reset_index()

    # rename column Month to Count
    df = df.rename(columns={"Month": "Count"})

    # Replace \n with "".
    df["y"] = df["y"].str.replace(" \n", "", regex=True)

    return df, rec


print(perform_analysis("csv"))
