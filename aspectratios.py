# aspectratios
import os
import cv2
import csv
import pandas as pd

# df = pd.DataFrame(columns=["path", "Aspect Ratio", "type (L/P)"])
# print(df)

paths = []
Ratios = []
Types = []

rootmap = "C:/Users\Ruben\Documents\Studie\Afstudeerproject\wikiart\wikiart"
for (_, root, _) in os.walk(rootmap):
    for folder in root:
        for (_,_,files) in os.walk(os.path.join(rootmap, folder)):
            for file in files:
                image_path = os.path.join(rootmap, folder, file)
                try:
                    img = cv2.imread(image_path)
                    ar = img.shape[1] / img.shape[0]
                    Ratios.append(ar)
                    paths.append(os.path.join(folder + "/" + file))
                    if ar > 1:
                        Types.append("landscape")
                    else:
                        Types.append("portrait")
                except:
                    pass
                
            
        break

# print(paths)
df = pd.DataFrame(list(zip(paths, Ratios, Types)), columns=["path", "Aspect Ratio", "type (L/P)"])
df.to_csv('C:/Users\Ruben\Documents\Studie\Afstudeerproject\Code\Aspectratio.csv')