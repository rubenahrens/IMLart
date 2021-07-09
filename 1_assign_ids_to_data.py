# aspectratios
import os
from typing_extensions import Annotated
import cv2
import csv
import pandas as pd

# df = pd.DataFrame(columns=["path", "Aspect Ratio", "type (L/P)"])
# print(df)

paths = []
ids = []
Annotated_ = []

count=0
rootmap = "C:/Users\Ruben\Documents\Studie\Afstudeerproject\wikiart\wikiart"
for (_, root, _) in os.walk(rootmap):
    for folder in root:
        for (_,_,files) in os.walk(os.path.join(rootmap, folder)):
            for file in files:
                image_path = os.path.join(rootmap, folder, file)
                paths.append(os.path.join(folder + "/" + file))
                ids.append(count)
                count += 1

# print(paths)
df = pd.DataFrame(list(paths), columns=["path"])
df.set_index('id')
df.to_csv('C:/Users\Ruben\Documents\Studie\Afstudeerproject\Code\id.csv')