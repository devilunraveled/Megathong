from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

analyzer = SentimentIntensityAnalyzer()

def polarityScores(text):
    return analyzer.polarity_scores(text)

def polarity(text):
    scores = polarityScores(text)
    print(scores)
    posWeight = 0.8
    neuWeight = 0.6
    negWeight = 1
    maxSc = np.argmax([posWeight * scores['pos'], negWeight * scores['neg'], neuWeight * scores['neu']])
    if maxSc == 0:
        return 'Positive'
    elif maxSc == 1:
        return 'Negative'
    else:
        return 'Neutral'
    
if __name__ == '__main__':
    text = input('Enter text: ')
    print(f'Polarity: {polarity(text)}')