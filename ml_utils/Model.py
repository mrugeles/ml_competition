import pandas as pd
import numpy as np

from keras.layers import Conv1D, MaxPooling1D, Embedding, Input

class Model():

    def create_embedding_index(self, path):
        embeddings_index = {}
        f = open(path)
        next(f)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def create_embedding_matrix(self, word_index, embedding_dim, embeddings_index):
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def create_model(self, word_index, embedding_matrix, embedding_dim, max_sequence_length):
        embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
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


        preds = Dense(len(labels[1]), activation='softmax')(x)

        model = Model(sequence_input, preds)
        return model

    def train_model(self, model_path):
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