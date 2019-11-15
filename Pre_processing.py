
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# Load the data
out_list = []
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run2.npy")

sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']
rep_list = [3690, 3690]

num_train = 3
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state

def outlier_remove(Sub_run_name, c0, ocm0):  # input subject and run name
    count = 0
    ocm0_new = np.zeros(np.shape(ocm0))
    if Sub_run_name is 's1r1':
        for p in range(0, c0):
            if ocm0[-1, p] > -200:
                ocm0_new[:, count] = ocm0[:, p]
                count = count + 1

    elif Sub_run_name is 's1r2':
        for p in range(0, c0):
            if ocm0[0, p] < 3000 and ocm0[-1, p] < 3000:
                    if ocm0[54, p] < 3000:
                        ocm0_new[:, count] = ocm0[:, p]
                        count = count + 1

    elif Sub_run_name is 's2r1':
        for p in range(0, c0):
            if ocm0[0, p] < 4000 and ocm0[-1, p] < 4500:
                ocm0_new[:, count] = ocm0[:, p]
                count = count + 1

    elif Sub_run_name is 's2r2':
        for p in range(0, c0):
            if ocm0[-1, p] < 1500:  # are we sure we need this?
            #if ocm0[-1, p] < 15000:  # it turned out that no cleaning works better for this case
                ocm0_new[:, count] = ocm0[:, p]
                count = count + 1

    elif Sub_run_name is 's3r1':
        for p in range(0, c0):
            if ocm0[-1, p] < 3000 and ocm0[40, p] < 4500:
                ocm0_new[:, count] = ocm0[:, p]
                count = count + 1

    elif Sub_run_name is 's3r2':
        ocm0_new = ocm0
        count = ocm0_new.shape[1]

    ocm0_new = ocm0_new[:, 0:count]
    return ocm0_new


def preprocessing(fidx):
    Sub_run_name = sr_list[fidx]
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    #crop data
    ocm = ocm[300:650, :]
    s, t = np.shape(ocm) # s=# of samples per trace. t=# of total traces

    # divide the data into each OCM and store absolute value
    b = np.linspace(0, t - 1, t)
    b0 = np.mod(b, 4) == 0
    ocm0 = ocm[:, b0]
    b1 = np.mod(b, 4) == 1
    ocm1 = ocm[:, b1]
    b2 = np.mod(b, 4) == 2
    ocm2 = ocm[:, b2]
    s, c0 = np.shape(ocm0)

    # first few traces are strange. Remove them
    ocm0 = ocm0[:, c0 - num_bh * rep_list[fidx]:]
    ocm1 = ocm1[:, c0 - num_bh * rep_list[fidx]:]
    ocm2 = ocm2[:, c0 - num_bh * rep_list[fidx]:]

    s, c0_new = np.shape(ocm0)
    t_sub = int(c0_new / num_bh)

    # Manually remove outlier. (OCM0 contains about 0.5% of outlier)
    ocm0_new = outlier_remove(Sub_run_name, c0_new, ocm0)
    s, c0_new_removed = np.shape(ocm0_new)
    t_sub_removed = int(c0_new_removed / num_bh)

    '''
    Skip filtering part for now
    '''

    return ocm0_new, ocm1, ocm2