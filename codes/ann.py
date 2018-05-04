import numpy as np
import pandas as pd

# import tensorflow as tf
from keras import backend as k
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop


def dataSetAnalysis(df):
    # view starting values of data set
    print("DataSet Head")
    print(df.head(3))
    print("=" * 30)
    # View features in data set
    print("Dataset Features Details")
    print(df.info())
    print("=" * 30)
    # view distribution of numerical features across the data set
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 30)
    # view distribution of categorical features across the data set
    # print("Dataset Categorical Features")
    # print(df.describe(include=['O']))
    # print("=" * 30)


def solve():
    # Read the data
    tr = pd.read_csv('../Dataset/train_all.csv', index_col='id')
    test = pd.read_csv('../Dataset/test_all.csv', index_col='id')
    # Make the class column contain 0 or 1 only
    tr['class'] = tr['class'] - 1
    # Initialize the X and the y
    X = tr.drop(['class'], axis=1)
    y = to_categorical(tr['class'].values, 2)

    ###### Build The Model #######
    model = Sequential()
    model.add(Dense(10, activation='sigmoid', input_shape=(X.shape[1],)))
    model.add(Dense(2, activation='softmax'))
    #### Compile the model ######
    model.compile(optimizer='adam', loss='categorical_hinge',  metrics=['accuracy'])
    model.fit(X, y, validation_split=0.2, verbose=1, epochs=30, callbacks=[EarlyStopping(patience=3)])
    # model.save('model.h5')
    # mod = load_model('model.h5')
    # predictions
    predictions = np.array(model.predict(test), dtype='int')[:, 1]
    # Display model summary
    model.summary()
    # Drop un needed Format
    test = test.drop(['B', 'G', 'R'], axis=1)
    test['class'] = (predictions + 1)
    # Save to csv
    test.to_csv('../output.csv', sep=',')


if __name__ == '__main__':
    dataset = pd.read_csv('../Dataset/train_all.csv')
    dataSetAnalysis(dataset)
    solve()
