import pandas as pd
from data_loader import load_raw_data

def split_data():
    
    df = load_raw_data()
    df=df.dropna(subset=['RainTomorrow'])


    year = pd.to_datetime(df['Date']).dt.year


    train_df = df[year < 2015]
    val_df   = df[year == 2015]
    test_df  = df[year > 2015]
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = split_data()
    print(f"Train set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")



