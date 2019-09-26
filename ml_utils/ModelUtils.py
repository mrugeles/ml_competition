import pandas as pd
import numpy as np

from keras.layers import Conv1D, MaxPooling1D, Embedding, Input
from keras.layers import GlobalMaxPooling1D, Dropout, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint  


class ModelUtils():

    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LENGTH = 1000

    def create_embedding_index(self, embedding_path):
        embeddings_index = {}
        f = open(embedding_path)
        next(f)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def create_embedding_matrix(self, word_index, embedding_path):
        embeddings_index = self.create_embedding_index(embedding_path)

        embedding_matrix = np.zeros((len(word_index) + 1, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def create_model(self, word_index, embedding_matrix, categories_size, max_sequence_length):
        embedding_layer = Embedding(len(word_index) + 1,
                            self.EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)

        sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        x = Conv1D(128, 5, kernel_initializer='glorot_normal', activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, kernel_initializer='glorot_normal', activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, kernel_initializer='glorot_normal', activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.01)(x)
        x = Dense(128, activation='relu')(x)


        preds = Dense(categories_size, activation='softmax')(x)

        model = Model(sequence_input, preds)
        return model

    def compile_fit(self, model, model_path, x_train, y_train, x_val, y_val):
        model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

        checkpointer = ModelCheckpoint(filepath=model_path, 
                                    verbose=0, save_best_only=True)


        model.fit(x_train, y_train,
                batch_size=1500,
                epochs=50,
                validation_data=(x_val, y_val),
                callbacks=[checkpointer])

    def train(self, x_train, y_train, x_val, y_val, word_index, embedding_path, model_path, max_sequence_length) :
        embedding_matrix = self.create_embedding_matrix(word_index, embedding_path)
        model = self.create_model(word_index, embedding_matrix, len(y_train[1]), max_sequence_length)
        self.compile_fit(model, model_path, x_train, y_train, x_val, y_val)
