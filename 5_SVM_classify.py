import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix

rootmap = "C:/Users/Ruben/Documents/Studie/Afstudeerproject/wikiart/wikiart"
x_train = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/x_train.csv")
x_test = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/x_test.csv")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
features = np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/image_1d_train.npy')
# print(len(features))
X = np.array(features)
y = np.array(x_train['Annotation'].to_list())
print(len(y))

from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
print("Training...")
clf.fit(X, y)

features = np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/image_1d_test.npy')
print("Testing...")
predictions = clf.predict(np.array(features))
accuracy = accuracy_score(x_test["Annotation"].to_numpy(), predictions)
print("Accuracy:", accuracy_score(x_test["Annotation"].to_numpy(), predictions))
print("f1-score:", f1_score(x_test["Annotation"].to_numpy(), predictions))
print("matrix:", multilabel_confusion_matrix(x_test["Annotation"].to_numpy(), predictions))