# python script
# -*- coding: utf-8 LF
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
@create: 2023/1/7-12:06
"""
import os
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from neurora.rdm_cal import eegRDM


# [n_cons, sub_0, n_trials, n_features, tp_0].
test_mat = scipy.io.loadmat(r'G:\data\project_data\test_mat.mat')['nbNodes_beta_4d']
data_4d = np.expand_dims(test_mat, axis=-1)  # [n_cons, n_trials, n_features, tp_0].
data_5d = np.expand_dims(data_4d, axis=1)

sub_rdms = eegRDM(data_5d, sub_opt=1, chl_opt=0,
                  time_opt=1, time_win=1, time_step=1,
                  method="correlation", abs=False)


matlab_rdm = scipy.io.loadmat(r'G:\data\project_data\test_rdm.mat')['RDM_nbNodes_beta']

plt.imshow(np.squeeze(matlab_rdm - sub_rdms))

(matlab_rdm - sub_rdms).max()


# data_avged = np.mean(np.squeeze(data_5d[:, 0, :, :, 0]), axis=1)  # (20 avged samples for each cate, n_vertices)
# np_rdm = 1 - np.corrcoef(data_avged, rowvar=True)  # (20 categories, n_vertices)
# # Each row of x represents a variable, and each column a single observation of all those variables.
# # we are going to calculate the
# (np_rdm - sub_rdms).max()
#
# (np_rdm - matlab_rdm).max()
