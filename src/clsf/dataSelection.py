import pandas as pd
import numpy as np
import os

neutralDfs = 25
trainSize = 1000
testSize = 250

files = os.listdir('data/redditComments')

addictionDf = pd.DataFrame()
for file in files:
    if not file.startswith('addiction'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    addictionDf = pd.concat([addictionDf, df])

adhdDf = pd.DataFrame()
for file in files:
    if not file.startswith('adhd'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    adhdDf = pd.concat([adhdDf, df])

anxietyDf = pd.DataFrame()
for file in files:
    if not file.startswith('anxiety'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    anxietyDf = pd.concat([anxietyDf, df])

bipolarDf = pd.DataFrame()
for file in files:
    if not file.startswith('bipolarreddit') and not file.startswith('bpd'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    bipolarDf = pd.concat([bipolarDf, df])
    bipolarDf['subreddit'] = 'bipolar'

depressionDf = pd.DataFrame()
for file in files:
    if not file.startswith('depression'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    depressionDf = pd.concat([depressionDf, df])

ptsdDf = pd.DataFrame()
for file in files:
    if not file.startswith('ptsd'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    ptsdDf = pd.concat([ptsdDf, df])

suicideDf = pd.DataFrame()
for file in files:
    if not file.startswith('suicidewatch'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    suicideDf = pd.concat([suicideDf, df])

neutralDf = pd.DataFrame()
neutralFiles = np.random.choice(files, neutralDfs)
for file in neutralFiles:
    df = pd.read_csv('data/redditComments/' + file)
    neutralDf = pd.concat([neutralDf, df])
    neutralDf['subreddit'] = 'neutral'


addictionDf = addictionDf.sample(trainSize + testSize)
adhdDf = adhdDf.sample(trainSize + testSize)
anxietyDf = anxietyDf.sample(trainSize + testSize)
bipolarDf = bipolarDf.sample(trainSize + testSize)
depressionDf = depressionDf.sample(trainSize + testSize)
ptsdDf = ptsdDf.sample(trainSize + testSize)
suicideDf = suicideDf.sample(trainSize + testSize)
neutralDf = neutralDf.sample(trainSize + testSize)

trainData = pd.concat([addictionDf[:trainSize], adhdDf[:trainSize], anxietyDf[:trainSize], bipolarDf[:trainSize], depressionDf[:trainSize], ptsdDf[:trainSize], suicideDf[:trainSize], neutralDf[:trainSize]])
testData = pd.concat([addictionDf[trainSize:], adhdDf[trainSize:], anxietyDf[trainSize:], bipolarDf[trainSize:], depressionDf[trainSize:], ptsdDf[trainSize:], suicideDf[trainSize:], neutralDf[trainSize:]])

print(trainData.groupby('subreddit').size())
print(testData.groupby('subreddit').size())

trainData.to_csv('data/labelListTrainData.csv', index=False)
testData.to_csv('data/labelListTestData.csv', index=False)