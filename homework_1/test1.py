import numpy as np
import pandas as pd
from time import time
from IPython.display import display
import visuals as vs
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt


data = pd.read_csv("census.csv")
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
features_final = pd.get_dummies(features_raw)
features_final = (features_final - features_final.min()) / (features_final.max() - features_final.min())
income = []
for s in income_raw:
    if s == ">50K":
        income.append(1)
    else:
        income.append(0)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(features_final, income, test_size=0.2, random_state=0)
# print(X_train, X_test)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


model_bes = GaussianNB()
model_bes.fit(X_train, Y_train)
y1_score = model_bes.predict_proba(X_test)

model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)
y2_score = model_lr.predict_proba(X_test)



model_dfc = DecisionTreeClassifier(max_depth=10)
model_dfc.fit(X_train, Y_train)
y3_score = model_dfc.predict_proba(X_test)

y1_pre = model_bes.predict(X_test)
y2_pre = model_lr.predict(X_test)
y3_pre = model_dfc.predict(X_test)
# print(y1_score, y2_score, y3_score)
from sklearn import metrics
import pylab as plt


def ks(y_predicted1,  y_predicted2,  y_predicted3, y_true):
    Font = {'size': 18, 'family': 'Times New Roman'}

    label1 = label2 = label3 = y_true
    fpr1, tpr1, thres1 = metrics.roc_curve(label1, y_predicted1[:, 1])
    fpr2, tpr2, thres2 = metrics.roc_curve(label2, y_predicted2[:, 1])
    fpr3, tpr3, thres3 = metrics.roc_curve(label3, y_predicted3[:, 1])
    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    roc_auc3 = metrics.auc(fpr3, tpr3)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr1, tpr1, 'b', label='bayes auc = %0.3f' % roc_auc1, color='Red')
    plt.plot(fpr2, tpr2, 'b', label='lr auc = %0.3f' % roc_auc2, color='k')
    plt.plot(fpr3, tpr3, 'b', label='dfc auc = %0.3f' % roc_auc3, color='RoyalBlue')
    plt.legend(loc='lower right', prop=Font)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=15)
    plt.show()
    return abs(fpr1 - tpr1).max(), abs(fpr2 - tpr2).max(), abs(fpr3 - tpr3).max()
# print(y1_score, y2_score)

print(ks(y1_score,  y2_score,  y3_score, Y_test))