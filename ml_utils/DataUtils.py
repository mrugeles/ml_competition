import pandas as pd
import numpy as np
from sklearn import preprocessing

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class DataUtils():

    MAX_NUM_WORDS = 20000
    MAX_SEQUENCE_LENGTH = 1000
    VALIDATION_SPLIT = 0.2

    def get_data(self, df, label_quality, lang):
        return df[(df['language'] == lang) & (df['label_quality'] == label_quality)]
    
    def filter_reliable_categories(self, df):
        feature = df['category'].value_counts().to_frame(name = 'counts')
        reliable_categories = list(set(feature[feature['counts'] >= 50].index.values))
        df = df[df['category'].isin(reliable_categories)]

        return df
    
    def encode_dataset(self, df):
        le = preprocessing.LabelEncoder()
        le.fit(df['category'])
        df['category_code'] = le.transform(df['category']) 
        
        texts = list(df['title'].values)
        labels = list(df['category_code'].values)

        tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        labels = to_categorical(np.asarray(labels))


        return data, labels, tokenizer.word_index

    def split_dataset(self, data, labels):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        num_validation_samples = int(self.VALIDATION_SPLIT * data.shape[0])

        x_train = data[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = data[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]

        return x_train, y_train, x_val, y_val

    def get_training_dataset(self, dataset, label_quality, lang):
        dataset = self.get_data(dataset, label_quality, lang)
        dataset = self.filter_reliable_categories(dataset)

        data, labels, word_index = self.encode_dataset(dataset)



        return self.split_dataset(data, labels), word_index
