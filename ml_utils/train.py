import sys
import pandas as pd

from ml_utils.DataUtils import DataUtils
from ml_utils.ModelUtils import ModelUtils

def main():
    if len(sys.argv) == 2:
        lang = sys.argv[1]
        dataUtils = DataUtils()
        modelUtils = ModelUtils()
        dataset = pd.read_csv('data/train.csv')

        x_train, y_train, x_val, y_val, word_index = dataUtils.get_training_dataset(dataset, 'reliable', lang)

        modelUtils.train(x_train, y_train, x_val, y_val, word_index, 'embeddings-m-model.vec', f'model_{lang}.hdf5')

if __name__ == '__main__':
    main()