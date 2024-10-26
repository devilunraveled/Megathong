import pandas as pd
from config import Paths

trainData = pd.read_csv(f"{Paths.data}/train.csv")[:21000]
testData = pd.read_csv(f"{Paths.data}/test.csv")[:3000]
evalData = pd.read_csv(f"{Paths.data}/validation.csv")[:1000]

trainData.to_csv(f"{Paths.data}/train.csv", index=False)
testData.to_csv(f"{Paths.data}/test.csv", index=False)
evalData.to_csv(f"{Paths.data}/validation.csv", index=False)
