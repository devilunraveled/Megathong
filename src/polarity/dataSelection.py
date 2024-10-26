import pandas as pd
import numpy as np
import os

files = os.listdir('data/redditComments')

dataDf = pd.DataFrame(columns=['post', 'sent_neg', 'sent_neu', 'sent_pos'])
for file in files:
    df = pd.read_csv('data/redditComments/' + file)
    df = df[['post', 'sent_neg', 'sent_neu', 'sent_pos']]
    negDf = df[np.argmax(df[['sent_neg', 'sent_neu', 'sent_pos']].values, axis=1) == 0]
    neuDf = df[np.argmax(df[['sent_neg', 'sent_neu', 'sent_pos']].values, axis=1) == 1]
    posDf = df[np.argmax(df[['sent_neg', 'sent_neu', 'sent_pos']].values, axis=1) == 2]

    if negDf.shape[0] > 20:
        negDf = negDf.sample(20)
    if neuDf.shape[0] > 10:
        neuDf = neuDf.sample(10)
    if posDf.shape[0] > 20:
        posDf = posDf.sample(20)

    dataDf = pd.concat([dataDf, negDf, neuDf, posDf])
    print(dataDf.shape)

dataDf.to_csv('data/polarityData.csv', index=False)