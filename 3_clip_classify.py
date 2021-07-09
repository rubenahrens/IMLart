import torch
import clip
from PIL import Image
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def classify_CLIP(path, classes, model, preprocess, device):

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    text = clip.tokenize(classes).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        problist = probs.flatten().tolist()
        print(problist)
    return problist.index(max(problist))
    
df = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/Annotation.csv")
rootmap = "C:/Users/Ruben/Documents/Studie/Afstudeerproject/wikiart/wikiart"

clipmap = []
clip_correct = []
classes= ["Portrait", "Landscape"]

for index, row in df.iterrows():
    if row['Annotation'] in (0,1):
        image_path = rootmap + "/" + row['path']
        clip_prediction = classify_CLIP(image_path, classes, model, preprocess, device)
        clipmap.append(clip_prediction)
    else:
        df.drop(index, inplace=True)
        


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(clipmap, df["Annotation"].to_numpy())
print("OG CLIP accuracy =", accuracy)