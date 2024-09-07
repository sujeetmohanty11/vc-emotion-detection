import numpy as np
import pandas as pd

import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import logging

logger = logging.getLogger('Data Preprocessing Logger')
logger.setLevel('DEBUG')
file_handler = logging.FileHandler('model.log')
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

#data
def load_data():
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    logger.info('Data fetched from data/raw')
    return train_data, test_data


#transformation
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    logger.info('Converted to Lowercase')
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    logger.info('Stop words removed')
    df.content=df.content.apply(lambda content : removing_numbers(content))
    logger.info('Numbers removed')
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    logger.info('Removed punctuations and whitespace')
    df.content=df.content.apply(lambda content : removing_urls(content))
    logger.info('Removed url')
    df.content=df.content.apply(lambda content : lemmatization(content))
    logger.info('lemmatization')
    return df

def save_data(train_processed_data, test_processed_data):
    data_path = os.path.join("data", "processed")
    os.makedirs(data_path)
    train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'))
    test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'))
    logger.info('Data saved in data/processed')


def main():
    train_data, test_data = load_data()
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    save_data(train_processed_data, test_processed_data)

if __name__ == '__main__':
    main()