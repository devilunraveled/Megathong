import pandas as pd
import os

files = os.listdir('data/redditComments')

addictionDf = pd.DataFrame()
for file in files:
    if not file.startswith('addiction'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    addictionDf = pd.concat([addictionDf, df])

addictionDf = addictionDf.groupby('author').filter(lambda x: len(x) > 2).sort_values(['author', 'date'])

adhdDf = pd.DataFrame()
for file in files:
    if not file.startswith('adhd'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    adhdDf = pd.concat([adhdDf, df])

adhdDf = adhdDf.groupby('author').filter(lambda x: len(x) > 2).sort_values(['author', 'date'])

anxietyDf = pd.DataFrame()
for file in files:
    if not file.startswith('anxiety'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    anxietyDf = pd.concat([anxietyDf, df])

anxietyDf = anxietyDf.groupby('author').filter(lambda x: len(x) > 2).sort_values(['author', 'date'])

bipolarDf = pd.DataFrame()
for file in files:
    if not file.startswith('bipolarreddit') and not file.startswith('bpd'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    bipolarDf = pd.concat([bipolarDf, df])

bipolarDf = bipolarDf.groupby('author').filter(lambda x: len(x) > 2).sort_values(['author', 'date'])

depressionDf = pd.DataFrame()
for file in files:
    if not file.startswith('depression'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    depressionDf = pd.concat([depressionDf, df])

depressionDf = depressionDf.groupby('author').filter(lambda x: len(x) > 2).sort_values(['author', 'date'])

ptsdDf = pd.DataFrame()
for file in files:
    if not file.startswith('ptsd'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    ptsdDf = pd.concat([ptsdDf, df])

ptsdDf = ptsdDf.groupby('author').filter(lambda x: len(x) > 2).sort_values(['author', 'date'])

suicideDf = pd.DataFrame()
for file in files:
    if not file.startswith('suicidewatch'):
        continue
    df = pd.read_csv('data/redditComments/' + file)
    suicideDf = pd.concat([suicideDf, df])

suicideDf = suicideDf.groupby('author').filter(lambda x: len(x) > 2).sort_values(['author', 'date'])

timeSeriesDf = pd.concat([addictionDf, adhdDf, anxietyDf, bipolarDf, depressionDf, ptsdDf, suicideDf])
timeSeriesDf.to_csv('data/timeSeriesData.csv', index=False)