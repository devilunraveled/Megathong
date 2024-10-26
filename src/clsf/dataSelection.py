import pandas as pd
import numpy as np
import os

neutralDfs = 20
dfSamples = 150

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

dataDf = pd.concat([addictionDf.sample(dfSamples), adhdDf.sample(dfSamples), anxietyDf.sample(dfSamples), bipolarDf.sample(dfSamples), depressionDf.sample(dfSamples), ptsdDf.sample(dfSamples), suicideDf.sample(dfSamples), neutralDf.sample(dfSamples)])

dataDf.to_csv('data/labelListData.csv', index=False)