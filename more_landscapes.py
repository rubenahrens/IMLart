import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
rootmap = "C:/Users/Ruben/Documents/Studie/Afstudeerproject/wikiart/wikiart"
x_train = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/CLIP_train.csv")
annotations = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/Annotation.csv")
all_csv = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/id.csv")

features = np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/image_1d_train.npy')
X = np.array(features)
y = np.array(x_train['Annotation'].to_list())

from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
print("Training...")
clf.fit(X, y)

samples = (random.sample(range(len(all_csv)), 100))
reccomendations = []
for sample in samples:
    if sample not in annotations["id"].to_list():
        image_path = rootmap + "/" + all_csv['path'][sample]
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features.cpu().numpy().flatten()
            prediction = clf.predict(np.array([image_features]))[0]
            if prediction == 1:
                reccomendations.append(sample)

np.save('annotation_reccomendation.npy', reccomendations)
