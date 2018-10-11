import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

# Loading the dataset
irisdataset = datasets.load_iris()

# getting the data and response of the dataset
x = irisdataset.data
y = irisdataset.target

from sklearn.svm import SVC
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
# Create a SVC classifier using an RBF kernel
svm = SVC(kernel='rbf', random_state=0, degree=4, gamma=1, C=1)
svm.fit(X_train, y_train)
print("kernel='rbf', random_state=0, degree=4, gamma=1, C=1")
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

# Test different settings in the degree and gamma
svm = SVC(kernel='rbf', random_state=0, degree=4, gamma=.5, C=1)
svm.fit(X_train, y_train)
print("Using SVC: kernel='rbf', random_state=0, degree=4, gamma=.5, C=1")
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

# Test different settings in the degree and gamma
svm = SVC(kernel='rbf', random_state=0, degree=4, gamma=.01, C=1)
svm.fit(X_train, y_train)
print("Using SVC: kernel='rbf', random_state=0, degree=4, gamma=.01, C=1")
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

# Test different settings in the degree and gamma
svm = SVC(kernel='rbf', random_state=0, degree=4, gamma=1, C=.5)
svm.fit(X_train, y_train)
print("Using SVC: kernel='rbf', random_state=0, degree=4, gamma=1, C=.5")
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

# Test different settings in the degree and gamma
svm = SVC(kernel='rbf', random_state=0, degree=4, gamma=1, C=.1)
svm.fit(X_train, y_train)
print("kernel='rbf', random_state=0, degree=4, gamma=1, C=.1")
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))