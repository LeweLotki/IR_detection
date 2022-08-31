import numpy as np
import pandas as pd
import json

path = 'E:\\IR_detection\\IR_detection\\IR_detection\\data\\test\\index.json'
with open(path) as json_data:
    data = json.load(json_data)
df = pd.DataFrame(data['frames'])
df = df[['videoMetadata', 'annotations']]
print(df.loc[[0]]['annotations'][0][0]['labels'])
print(df.loc[[1]]['videoMetadata'][1])