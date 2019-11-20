## This code imports ROC parameters and visualize how it changed.


import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve


plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.size"] = 11

file_loc = "C:\\Users\\Kwon\\PycharmProjects\\PancOCM_Unsup\\hidden128_final64_epoch500_batch128_raw"

sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
FPR_my = 0.1

auc_score = np.zeros((6, 4, 5))  # 1st dim: fidx, 2nd dim: ocm_ch, 3rd dim: bh
TPR_my = np.zeros((6, 4, 5))

for fidx in range(0, 6, 1):
    Sub_run_name = sr_list[fidx]
    fig_auc, (ax_auc0, ax_auc1, ax_auc2, ax_auc3) = plt.subplots(1, 4, figsize=(16, 5))
    fig_tpr, (ax_tpr0, ax_tpr1, ax_tpr2, ax_tpr3) = plt.subplots(1, 4, figsize=(16, 5))
    x = np.linspace(6, 10, 5)

    for ocm_ch in range(0, 4): # ocm_num==4 is all ocm
        for bh in range(0, 5):
            fname = file_loc + '\\roc_data_'+Sub_run_name+'_ch'+str(ocm_ch)+'_bh'+str(bh)+'_raw.pkl'
            with open(fname, 'rb') as f:
                true_label, mse_score = pickle.load(f)

            fpr, tpr, thresholds = roc_curve(true_label, mse_score)
            auc_score[fidx, ocm_ch, bh] = roc_auc_score(true_label, mse_score)

            temp_arg = np.where(fpr < FPR_my)  # get argument where lower than FPR_my
            TPR_my[fidx, ocm_ch, bh] = tpr[len(temp_arg[0])]  # Get TPR corresponds to FPR_my

#plt.show()
for fidx in range(0, 6, 1):
    ax_auc0.plot(x, auc_score[fidx, 0, :], marker='o', label=sr_list[fidx]) # ocm0
    ax_auc1.plot(x, auc_score[fidx, 1, :], marker='o', label=sr_list[fidx]) # ocm1
    ax_auc2.plot(x, auc_score[fidx, 2, :], marker='o', label=sr_list[fidx]) # ocm2
    ax_auc3.plot(x, auc_score[fidx, 3, :], marker='o', label=sr_list[fidx]) # ocm3
    ax_tpr0.plot(x, TPR_my[fidx, 0, :], marker='o', label=sr_list[fidx]) # ocm0
    ax_tpr1.plot(x, TPR_my[fidx, 1, :], marker='o', label=sr_list[fidx]) # ocm1
    ax_tpr2.plot(x, TPR_my[fidx, 2, :], marker='o', label=sr_list[fidx]) # ocm2
    ax_tpr3.plot(x, TPR_my[fidx, 3, :], marker='o', label=sr_list[fidx]) # ocm3

ax_auc0.legend(loc = 'best')
ax_tpr0.legend(loc = 'best')

ax_auc0.set_title('AUC, OCM0')
ax_auc1.set_title('AUC, OCM1')
ax_auc2.set_title('AUC, OCM2')
ax_auc3.set_title('AUC, all OCM')
ax_tpr0.set_title('TPR when FPR is '+str(FPR_my)+', OCM0')
ax_tpr1.set_title('TPR when FPR is '+str(FPR_my)+', OCM1')
ax_tpr2.set_title('TPR when FPR is '+str(FPR_my)+', OCM2')
ax_tpr3.set_title('TPR when FPR is '+str(FPR_my)+', all OCM')

ax_auc0.set_xlabel('Breath-hold number')
ax_auc1.set_xlabel('Breath-hold number')
ax_auc2.set_xlabel('Breath-hold number')
ax_auc3.set_xlabel('Breath-hold number')
ax_tpr0.set_xlabel('Breath-hold number')
ax_tpr1.set_xlabel('Breath-hold number')
ax_tpr2.set_xlabel('Breath-hold number')
ax_tpr3.set_xlabel('Breath-hold number')

ax_auc0.set_ylabel('AUC')
ax_auc1.set_ylabel('AUC')
ax_auc2.set_ylabel('AUC')
ax_auc3.set_ylabel('AUC')
ax_tpr0.set_ylabel('TPR')
ax_tpr1.set_ylabel('TPR')
ax_tpr2.set_ylabel('TPR')
ax_tpr3.set_ylabel('TPR')

ax_auc0.set_ylim([0, 1])
ax_auc1.set_ylim([0, 1])
ax_auc2.set_ylim([0, 1])
ax_auc3.set_ylim([0, 1])
ax_tpr0.set_ylim([0, 1])
ax_tpr1.set_ylim([0, 1])
ax_tpr2.set_ylim([0, 1])
ax_tpr3.set_ylim([0, 1])

plt.xticks(np.arange(6, 11, 1))
#plt.show()

fig_auc.savefig('AUC.png')
fig_tpr.savefig('TPR.png')





