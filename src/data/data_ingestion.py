import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
import logging

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger('Data Ingestion Logger')
logger.setLevel('DEBUG')

file_handler = logging.FileHandler('model.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_params(path: str) -> float:
    try:
        test_size = yaml.safe_load(open(path, 'r'))['data_ingestion']['test_size']
        return test_size
    except:
        logger.error('File Not Found')


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    logger.info('Data Loaded')
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['tweet_id'])
    final_df = df[df['sentiment'].isin(['neutral','sadness'])]
    final_df['sentiment'] = final_df['sentiment'].replace({'neutral':1, 'sadness':0})
    logger.info('Data Preproceessed')
    return final_df

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    data_path = os.path.join("data", "raw")
    os.makedirs(data_path)
    train_data.to_csv(os.path.join(data_path, 'train.csv'))
    test_data.to_csv(os.path.join(data_path, 'test.csv'))
    logger.info('Data Saved')


def main():
    test_size = load_params('params.yaml')
    df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = preprocess_data(df)
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    save_data(train_data, test_data)


if __name__ == '__main__':
    main()