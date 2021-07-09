import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
rootmap = "C:/Users/Ruben/Documents/Studie/Afstudeerproject/wikiart/wikiart"
x_train = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/x_train.csv")

features = []
probsall = []
classes= ["Portrait", "Landscape"]

for index, row in x_train.iterrows():
    image_path = rootmap + "/" + row['path']
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(classes).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        features.append(image_features.cpu().numpy().flatten())
        
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        problist = probs.flatten()
        probsall.append(problist)

print(len(features))
np.save('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/image_1d_train.npy', np.array(features))
np.save('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/problist_train.npy', np.array(probsall))

x_test = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/x_test.csv")
features = []
for index, row in x_test.iterrows():
    image_path = rootmap + "/" + row['path']
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        feat = image_features.cpu().numpy().flatten()
        features.append(feat)

np.save('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/image_1d_test.npy', np.array(features))
