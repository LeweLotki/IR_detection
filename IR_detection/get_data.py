import numpy as np
import pandas as pd
import json

path = 'E:\\python_codes\\rumcajs\\data\\video_thermal_test\\index.json'
with open(path) as json_data:
    data = json.load(json_data)
df = pd.DataFrame(data['frames'])
df = df[['videoMetadata', 'annotations']]
print(df.loc[[1]]['annotations'][1][0]['labels'])