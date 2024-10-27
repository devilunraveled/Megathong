import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
# from src.clsf.config import RedditData

class RedditData:
    labels : list = ['addiction', 'anxiety', 'bipolar', 'depression', 'ptsd', 'adhd', 'suicidewatch', 'neutral']

with open('data/intensityHistory.pkl', 'rb') as f:
    intensityHistory = pkl.load(f)

with open('data/polarityHistory.pkl', 'rb') as f:
    polarityHistory = pkl.load(f)

intensityHistory = np.array(intensityHistory)
labels = sorted(list(RedditData.labels))
meanIntensities = np.mean(intensityHistory, axis=0)
maxIndices = np.argsort(meanIntensities)[-3:]
maxLabels = [labels[i] for i in maxIndices]
intensityHistory = intensityHistory[:, maxIndices]

plt.figure()
for i in range(3):
    plt.plot(intensityHistory[:, i], label=maxLabels[i])
plt.title('Intensity History')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.show()
