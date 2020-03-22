'''
Compute and report confusion matrices, mis-classification rates, and F1 score
for LDA and QDA on height and weight training data.

'''

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Prep Data

data = np.loadtxt('HeightWeight.txt', int, delimiter=',')
data_label = data[:,0]
data_x = data[:,[1,2]]

ntrain = 150
ntest = len(data) - ntrain

train_label = data_label[:ntrain]
test_label = data_label[ntrain:]
train_x = data_x[:ntrain]
test_x = data_x[ntrain:]

#LDA

lda = LinearDiscriminantAnalysis()
lda.fit(train_x,train_label)
lda_pred = lda.predict(test_x)
lda_accu = accuracy_score(test_label,lda_pred)
print('Mis-classification for LDA: ' + str((1 - lda_accu) * 100))
lda_cm = confusion_matrix(test_label,lda_pred)
print('LDA Confussion Matrix')
print(lda_cm)
lda_f1 = f1_score(test_label,lda_pred)
print('LDA F1 score: ' + str(lda_f1))

#QDA

qda = QuadraticDiscriminantAnalysis()
qda.fit(train_x,train_label)
qda_pred = lda.predict(test_x)
qda_accu = accuracy_score(test_label,qda_pred)
print('Mis-classification for QDA: ' + str((1 - qda_accu) * 100))
qda_cm = confusion_matrix(test_label,qda_pred)
print('QDA Confussion Matrix')
print(qda_cm)
qda_f1 = f1_score(test_label,qda_pred)
print('LDA F1 score: ' + str(qda_f1))
