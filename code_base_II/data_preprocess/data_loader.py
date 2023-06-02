import pandas as pd

class DataLoader:
    def load_dataset(self, filename):
        df = pd.read_csv(filename)
        return df

    def save_dataset(self, data, filename):
        data.to_csv(filename, index=False)
