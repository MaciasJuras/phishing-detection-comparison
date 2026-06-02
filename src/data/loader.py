import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str = "data/raw/Phishing_Email.csv"):
    if path == "data/raw/Phishing_Email.csv":
        fallback_path = "data/raw/PhisingEmail.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            df = pd.read_csv(fallback_path)
    else:
        df = pd.read_csv(path)

    # Rename to standard column names
    df = df.rename(columns={"Email Text": "body", "Email Type": "label"})
    df["label"] = df["label"].map({"Phishing Email": 1, "Safe Email": 0})

    # Drop rows with missing body or label
    df = df.dropna(subset=["body", "label"])
    df["body"] = df["body"].astype(str)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"[Loader] Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"[Loader] Phishing ratio (test): {test_df['label'].mean():.2%}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)