import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import seaborn as sns
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm


plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

sr_list = ['s1r1']

Sub_run = sr_list[0]
sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw

with open(sr_name, 'rb') as f:
    ocm0_all, ocm1_all, ocm2_all = pickle.load(f)

print(ocm0_all.shape)
d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
depth = ocm0_all.shape[0]
num_max = ocm0_all.shape[1]  # Total num of traces
bh = int(num_max // 5)

# Remove "10min after" component
ocm0_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"
ocm1_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"
ocm2_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"

ocm0_ba[:, 0:num_max] = ocm0_all[:, :, 0]  # add "before"
ocm1_ba[:, 0:num_max] = ocm1_all[:, :, 0]
ocm2_ba[:, 0:num_max] = ocm2_all[:, :, 0]
ocm0_ba[:, num_max:2 * num_max] = ocm0_all[:, :, 1]  # add "after"
ocm1_ba[:, num_max:2 * num_max] = ocm1_all[:, :, 1]
ocm2_ba[:, num_max:2 * num_max] = ocm2_all[:, :, 1]

# Add to one variable
ocm_ba = np.zeros([depth, 2 * num_max, 3])
ocm_ba[:, :, 0] = ocm0_ba[:, :]
ocm_ba[:, :, 1] = ocm1_ba[:, :]
ocm_ba[:, :, 2] = ocm2_ba[:, :]
print(ocm_ba.shape)

# Transpose
ocm_ba_t = np.einsum('abc->bac', ocm_ba)

bh_train = 1  # bh used for train
bh_test = 2  # bh used for test

X_train = np.zeros([bh, depth])
X_test = np.zeros([bh, depth])
X_train = ocm_ba_t[bh*(bh_train-1):bh*(bh_train), :, 0]
X_test = ocm_ba_t[bh*(bh_test-1):bh*(bh_test), :, 0]



# Apply standard scaler to output from resnet50
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=2, whiten=True)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Train classifier and obtain predictions for OC-SVM
oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

oc_svm_clf.fit(X_train)
if_clf.fit(X_train)

oc_svm_preds = oc_svm_clf.predict(X_test)
if_preds = if_clf.predict(X_test)

# Further compute accuracy, precision and recall for the two predictions sets obtained
