# sample 50k records from the dataset
import pandas as pd

def main():
    print('Sampling 50k records from the dataset... ', end='')
    df = pd.read_csv('../data/train.csv')
    df = df.sample(50000)
    df.to_csv('../data/train50k.csv', index=False)
    print('Done!')

if __name__ == "__main__":
    main()