import pandas as pd
from Intelligence.inference import analyze_sentiment


def perform_analysis(file_type: str, path: str = "sentiment_analysis.csv"):
    if file_type == "csv":
        # Loading CSV data
        data = pd.read_csv(f"../Data/{path}")
    else:
        return "Error! file type not supported."

    # New empty column for recommendation
    data["y"] = ""

    # Sentimental analysis.
    data["y"] = data["text"].apply(analyze_sentiment)

    # Results
    df = data.groupby("y")["Month"].count().reset_index()

    # rename column Month to Count
    df = df.rename(columns={"Month": "Count"})

    # Replace \n with "".
    df["y"] = df["y"].str.replace(" \n", "", regex=True)

    return df


# print(perform_analysis("csv"))
