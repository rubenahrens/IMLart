import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/Annotation.csv")

df = df[df.Annotation != 2]
x_train,x_test=train_test_split(df)
x_train.to_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/x_train.csv", index=False)
x_test.to_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/x_test.csv", index=False)