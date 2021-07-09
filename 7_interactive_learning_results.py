import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, squeeze=False)

IL0=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 alpha 5.npy')
IL1=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 20 alpha 5.npy')
IL2=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 50 alpha 5.npy')
IL3=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 querybycommittee alpha 5.npy')
IL4=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning random alpha 5.npy')



axs[0, 0].plot([np.mean(k) for k in zip(*IL0)])
axs[0, 0].plot([np.mean(k) for k in zip(*IL1)])
axs[0, 0].plot([np.mean(k) for k in zip(*IL2)])
axs[0, 0].plot([np.mean(k) for k in zip(*IL3)])
axs[0, 0].plot([np.mean(k) for k in zip(*IL4)])
# axs[0, 0].legend(["offset 0", "offset 20", "offset 50", "query by commmittee", "random sampling"])
axs[0, 0].set_title("Interactive machine learning performance on test set, alpha=5")

print("Accuracy: ",[np.mean(k) for k in zip(*IL0)][-1])
print("Accuracy: ",[np.mean(k) for k in zip(*IL1)][-1])
print("Accuracy: ",[np.mean(k) for k in zip(*IL2)][-1])
print("Accuracy: ",[np.mean(k) for k in zip(*IL3)][-1])
print("Accuracy: ",[np.mean(k) for k in zip(*IL4)][-1])

IL0=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 alpha 3.npy')
IL1=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 20 alpha 3.npy')
IL2=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 50 alpha 3.npy')
IL3=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 querybycommittee alpha 3.npy')
IL4=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning random alpha 3.npy')
print(IL4[0])

axs[0, 1].plot([np.mean(k) for k in zip(*IL0)])
axs[0, 1].plot([np.mean(k) for k in zip(*IL1)])
axs[0, 1].plot([np.mean(k) for k in zip(*IL2)])
axs[0, 1].plot([np.mean(k) for k in zip(*IL3)])
axs[0, 1].plot([np.mean(k) for k in zip(*IL4)])
# axs[0, 1].legend(["offset 0", "offset 20", "offset 50", "query by commmittee", "random sampling"])
axs[0, 1].set_title("Interactive machine learning performance on test set, alpha=3")

print("Accuracy: ",[np.mean(k) for k in zip(*IL0)][-1])
print("Accuracy: ",[np.mean(k) for k in zip(*IL1)][-1])
print("Accuracy: ",[np.mean(k) for k in zip(*IL2)][-1])
print("Accuracy: ",[np.mean(k) for k in zip(*IL3)][-1])
print("Accuracy: ",[np.mean(k) for k in zip(*IL4)][-1])

IL0=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 alpha 5 fit_set.npy')
IL1=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 20 alpha 5 fit_set.npy')
IL2=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 50 alpha 5 fit_set.npy')
IL3=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 querybycommittee alpha 5 fit_set.npy')
IL4=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning random alpha 5 fit_set.npy')


axs[1, 0].plot([np.mean(k) for k in zip(*IL0)])
axs[1, 0].plot([np.mean(k) for k in zip(*IL1)])
axs[1, 0].plot([np.mean(k) for k in zip(*IL2)])
axs[1, 0].plot([np.mean(k) for k in zip(*IL3)])
axs[1, 0].plot([np.mean(k) for k in zip(*IL4)])
# axs[1, 0].legend(["offset 0", "offset 20", "offset 50", "query by commmittee", "random sampling"])
axs[1, 0].set_title("Interactive machine learning performance on fit set set, alpha=5")

# print("Accuracy: ",[np.mean(k) for k in zip(*IL0)][-1])
# print("Accuracy: ",[np.mean(k) for k in zip(*IL1)][-1])
# print("Accuracy: ",[np.mean(k) for k in zip(*IL2)][-1])
# print("Accuracy: ",[np.mean(k) for k in zip(*IL3)][-1])
# print("Accuracy: ",[np.mean(k) for k in zip(*IL4)][-1])

IL0=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 alpha 3 fit_set.npy')
IL1=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 20 alpha 3 fit_set.npy')
IL2=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 50 alpha 3 fit_set.npy')
IL3=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 querybycommittee alpha 3 fit_set.npy')
IL4=np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning random alpha 3 fit_set.npy')

axs[1, 1].plot([np.mean(k) for k in zip(*IL0)])
axs[1, 1].plot([np.mean(k) for k in zip(*IL1)])
axs[1, 1].plot([np.mean(k) for k in zip(*IL2)])
axs[1, 1].plot([np.mean(k) for k in zip(*IL3)])
axs[1, 1].plot([np.mean(k) for k in zip(*IL4)])
axs[1, 1].legend(["US: offset 0", "US: offset 20", "US: offset 50", "query by commmittee", "random sampling"])
axs[1, 1].set_title("Interactive machine learning performance on fit set set, alpha=3")

for ax in axs.flat:
    ax.set(xlabel='Iteration', ylabel='Accuracy')
    


plt.show()

# print("Accuracy: ",[np.mean(k) for k in zip(*IL0)][-1])
# print("Accuracy: ",[np.mean(k) for k in zip(*IL1)][-1])
# print("Accuracy: ",[np.mean(k) for k in zip(*IL2)][-1])
# print("Accuracy: ",[np.mean(k) for k in zip(*IL3)][-1])
# print("Accuracy: ",[np.mean(k) for k in zip(*IL4)][-1])