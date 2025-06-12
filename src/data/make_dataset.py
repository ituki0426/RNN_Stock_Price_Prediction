import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def make_dataset():
    dataset_train = pd.read_csv('./data/Google_Stock_Price_Train.csv')
    scaler = MinMaxScaler(feature_range = (0, 1))
    train = dataset_train.loc[:, ["Open"]].values
    train_scaled = scaler.fit_transform(train)
    dataset_train = dataset_train.drop(labels="Date",axis=1)
    dataset_train["Close"] = dataset_train["Close"].str.replace(",", "").astype(float)
    dataset_train["Volume"] = dataset_train["Volume"].str.replace(",","").astype(float)
    scaler_dataset_train = MinMaxScaler(feature_range = (0, 1))
    scaled_dataset_train  = scaler_dataset_train.fit_transform(dataset_train)
    scaled_dataset_train = pd.DataFrame(
        scaled_dataset_train,
        columns=dataset_train.columns,
        index=dataset_train.index
        )
    # Creating a data structure with 50 timesteps and 1 output
    X_train_Open = []
    X_train_High = []
    X_train_Low = []
    X_train_Close = []
    y_train = []
    timesteps = 50
    for i in range(timesteps, 1258):
        X_train_Open.append(scaled_dataset_train["Open"][i-timesteps:i].values)
        X_train_High.append(scaled_dataset_train["High"][i-timesteps:i].values)
        X_train_Low.append(scaled_dataset_train["Low"][i-timesteps:i].values)
        X_train_Close.append(scaled_dataset_train["Close"][i-timesteps:i].values)
        y_train.append(train_scaled[i, 0])
        X_train_Open, X_train_High, X_train_Low, X_train_Close, y_train = np.array(X_train_Open), np.array(X_train_High), np.array(X_train_Low), np.array(X_train_Close), np.array(y_train)
    # Reshaping
    X_train_Open = np.reshape(X_train_Open, (X_train_Open.shape[0], X_train_Open.shape[1], 1))
    X_train_High = np.reshape(X_train_High, (X_train_High.shape[0], X_train_High.shape[1], 1))
    X_train_Low = np.reshape(X_train_Low, (X_train_Low.shape[0], X_train_Low.shape[1], 1))
    X_train_Close = np.reshape(X_train_Close, (X_train_Close.shape[0], X_train_Close.shape[1], 1))
    # Add your dataset creation logic here
    # For example, loading data, preprocessing, etc.
    X_train = {
        "Open": X_train_Open,
        "High": X_train_High,
        "Low": X_train_Low,
        "Close": X_train_Close
    }
    y_train = np.array(y_train)
    return X_train, y_train

if __name__ == "__main__":
    make_dataset()
    print("Dataset created successfully.")         