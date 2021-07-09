from numpy.core.fromnumeric import clip
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import random

rootmap = "C:/Users/Ruben/Documents/Studie/Afstudeerproject/wikiart/wikiart"
x_train = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/x_train.csv")
x_test = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/x_test.csv")
test_features = np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/image_1d_test.npy')
train_features = np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/image_1d_train.npy')

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = np.array(train_features)
y = np.array(x_train['Annotation'].to_list())
clip_probabilities = np.load('C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/problist_train.npy')

from sklearn.svm import SVC

def interactive_learning(X, y, test_annotations, offset=0, alpha=5, set="test_set", method="us"):
    """interactive machine learning:

    simulation of interactive machine learning.

    Parameters
    ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
            
        test_annotations: iterable
            Testing targets. Must fulfill label requirements for all steps of
            the pipeline.
            
        offset: int, default=0
            Determines how many samples are to be skipped when sampling the
            most uncertain samples. With 0 the aplha most uncertain samples
            will be added to fit_set.
        
        alpha: int, default=5
            Determines how many samples need to be added to the model
            every training cycle.
        
        set: {'test_set, fit_set'}, default="test_set"
            get accuracies from test or fit set.
        
        method: {'us', 'qbc', 'random'}, default="us"
            us: uncertainty sampling
            qbc: query by committee
            random: random sampling
        
    Returns
    -------
        accuracies : accuracies on set for every training cycle
    """

    # pick the five first images to train
    # if training set only contains one class, add data until it contains two
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    fit_set_x, fit_set_y = [[0], [0]]
    while fit_set_y.count(fit_set_y[0]) == len(fit_set_y):
        fit_set = random.sample([i for i in range(X.shape[0])], alpha)
        fit_set_x, fit_set_y = [X[i] for i in fit_set], [y[i] for i in fit_set]

    count = 0
    accuracies = []

    # train until every image has been added or until the algorithm has
    # a certainty of at least 80 percent for every image
    while count < X.shape[0] - alpha:
        # fit model on training data: training
        clf.fit(fit_set_x, fit_set_y)
        
        # uncertainty sampling
        if method == "us":
            # predict class and get the probabilities for both classes with
            # predict_proba
            predictions = clf.predict_proba(X)
            
            # the closer a probability is to 0.5, the more uncertain
            # the prediction
            certainty = np.abs(predictions[:,1] - 0.5)
            
            # take the alpha most uncertain predictions and add the data of those images to the training set
            # with unsimulated interactive learning the labels would have been annotated by the user
            worst_predictions = np.argsort(certainty)
            count_pred = 0
            for predi in worst_predictions:
                if predi not in fit_set:
                    if offset -1 < count_pred < offset + alpha:
                        fit_set_x = np.append(fit_set_x, [X[predi]], axis=0)
                        fit_set_y = np.append(fit_set_y, [y[predi]], axis=0)
                    count_pred += 1
        # query by comittee
        elif method == "qbc":
            # predict class and get the probabilities for both classes with
            # predict_proba
            predictions = clf.predict_proba(X)
            # clip_probabilities = np.load('problist_train.npy')
            
            disagree = np.abs(clip_probabilities[:,1]-predictions[:,1])*-1
            
            # take the alpha most disagreed by clip predictions and add the data of those images to the training set
            # with unsimulated interactive learning the labels would have been annotated by the user
            # worst_predictions = np.argsort(disagree)[count+offset:count+offset+5]
            worst_predictions = np.argsort(disagree)
            count_pred = 0
            for predi in worst_predictions:
                if predi not in fit_set:
                    if count_pred < alpha:
                        fit_set.append(predi)
                        fit_set_x = np.append(fit_set_x, [X[predi]], axis=0)
                        fit_set_y = np.append(fit_set_y, [y[predi]], axis=0)
                        count_pred += 1

        # random sampling
        elif method == "random":
            worst_predictions = random.sample([i for i in range(X.shape[0]) if i not in fit_set], min(alpha, (X.shape[0] - len(fit_set))))
            for predi in worst_predictions:
                fit_set.append(predi)
                fit_set_x = np.append(fit_set_x, [X[predi]], axis=0)
                fit_set_y = np.append(fit_set_y, [y[predi]], axis=0)

        if set == "test_set":
            predictions = clf.predict(np.array(test_features))
            accuracies.append(accuracy_score(test_annotations, predictions))
        if set == "fit_set":
            predictions = clf.predict(np.array(fit_set_x))
            accuracies.append(accuracy_score(fit_set_y, predictions))
        count += alpha
    return accuracies

# # 5 experiments
sample_size = 1

# # alpha 5
# # test set accuracies
# IL = [interactive_learning(X, y, x_test["Annotation"].to_numpy()) for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 alpha 5", np.array(IL))

# IL1 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), offset=20) for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 20 alpha 5", np.array(IL1))

# IL2 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), offset=50) for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 50 alpha 5", np.array(IL2))

# IL3 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), method="qbc") for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 querybycommittee alpha 5", np.array(IL3))

# IL4 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), method="random") for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning random alpha 5", np.array(IL4))


# # alpha 3
# # test set accuracies
# IL = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), alpha=3) for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 alpha 3", np.array(IL))

IL1 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), offset=20, alpha=3) for i in range(sample_size)]
np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 20 alpha 3", np.array(IL1))

# IL2 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), offset=50, alpha=3) for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 50 alpha 3", np.array(IL2))

# IL3 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), alpha=3, method="qbc") for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 querybycommittee alpha 3", np.array(IL3))

# IL4 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), alpha=3, method="random") for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning random alpha 3", np.array(IL4))

# # alpha 5
# # fit set accuracies
# IL = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), set="fit_set") for i in range(sample_size)]
# IL1 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), offset=20, set="fit_set") for i in range(sample_size)]
# IL2 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), offset=50, set="fit_set") for i in range(sample_size)]
# IL3 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), method="qbc", set="fit_set") for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 querybycommittee alpha 5 fit_set", np.array(IL3))
# IL4 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), method="random", set="fit_set") for i in range(sample_size)]

# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 alpha 5 fit_set", np.array(IL))
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 20 alpha 5 fit_set", np.array(IL1))
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 50 alpha 5 fit_set", np.array(IL2))
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning random alpha 5 fit_set", np.array(IL4))

# # alpha 3
# # fit set accuracies
# IL = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), alpha=3, set="fit_set") for i in range(sample_size)]
# IL1 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), offset=20, alpha=3, set="fit_set") for i in range(sample_size)]
# IL2 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), offset=50, alpha=3, set="fit_set") for i in range(sample_size)]
# IL3 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), alpha=3, method="qbc", set="fit_set") for i in range(sample_size)]
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 querybycommittee alpha 3 fit_set", np.array(IL3))
# IL4 = [interactive_learning(X, y, x_test["Annotation"].to_numpy(), alpha=3, method="random", set="fit_set") for i in range(sample_size)]

# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 0 alpha 3 fit_set", np.array(IL))
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 20 alpha 3 fit_set", np.array(IL1))
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning results offset 50 alpha 3 fit_set", np.array(IL2))
# np.save("C:/Users/Ruben/Documents/Studie/Afstudeerproject/IL_Results/Interactive learning random alpha 3 fit_set", np.array(IL4))