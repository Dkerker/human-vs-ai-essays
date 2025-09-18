import pandas as pd

def load_data(path = "data/AI_Human.csv"):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.columns)

