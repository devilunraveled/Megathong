from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

analyzer = SentimentIntensityAnalyzer()

def polarityScores(text):
    return analyzer.polarity_scores(text)

def polarity(text):
    scores = polarityScores(text)
    maxSc = np.argmax([scores['pos'], scores['neg'], scores['neu']])
    if maxSc == 0:
        return 'Positive'
    elif maxSc == 1:
        return 'Negative'
    else:
        return 'Neutral'
    
if __name__ == '__main__':
    text = input('Enter text: ')
    print(f'Polarity: {polarity(text)}')