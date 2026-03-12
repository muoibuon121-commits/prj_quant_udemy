import numpy as np
import yfinance as download_data

def prepare_data(symbol, seq_length=60):
    df = download_data.download(symbol, start="2020-01-01")
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=21).std() # Target
    df = df.dropna()
    data = df[['Returns', 'Volatility']].values # Có thể thêm Volume, RSI..
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length]) # 60 ngày quá khứ
        y.append(data[i + seq_length, 1]) # Biến động ngày thứ 61
    return torch.tensor(X).float(), torch.tensor(y).float()
