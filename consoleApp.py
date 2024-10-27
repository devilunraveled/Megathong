from src.polarity.polarity import polarity
from src.clsf.classify import classify
from src.clsf.config import RedditData
from src.keywordExtraction.keywordExtraction import getKeyPhrases
import numpy as np
import pickle as pkl

if __name__ == '__main__':
    ask = True
    intensityHistory = []
    polarityHistory = []
    while ask:
        text = input('Enter text: ')
        pol = polarity(text)
        print(f'Polarity: {pol}')
        intensities = classify(text)
        labels = sorted(list(RedditData.labels))
        maxLabel = labels[np.argmax(intensities)]
        print(f'Category: {maxLabel}')
        print(f'Intensity Scores:')
        for i in range(len(intensities)):
            print(f'{labels[i]} - {intensities[i]}') 
        keyPhrases = getKeyPhrases(text)
        keyPhrases = keyPhrases[:min(3, len(keyPhrases))]
        print(f'Concerns :')
        for keyPhrase in keyPhrases:
            print(keyPhrase)
        intensityHistory.append(intensities)
        polarityHistory.append(pol)
        ask = input('Do you want to continue? (y/n) ') == 'y'

    with open('data/intensityHistory.pkl', 'wb') as f:
        pkl.dump(intensityHistory, f)

    with open('data/polarityHistory.pkl', 'wb') as f:
        pkl.dump(polarityHistory, f)