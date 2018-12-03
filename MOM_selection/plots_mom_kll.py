##Pickle dump
import pickle

filename = 'outliers_plus_heavytail_range.pickle'	
with open(filename, 'rb') as f:
	outliers_plus_heavytail_range = pickle.load(f)

filename = 'selected_estimators_errors_all.pickle'
with open(filename, 'rb') as f:
	selected_estimators_errors_all = pickle.load(f)

filename = 'nb_outliers1_in_selected_blocks_all.pickle'
with open(filename, 'rb') as f:
	nb_outliers1_in_selected_blocks_all = pickle.load(f)

filename = 'nb_outliers1_in_best_estimator_subsample_all.pickle'
with open(filename, 'rb') as f:
	nb_outliers1_in_best_estimator_subsample_all = pickle.load(f)

filename = 'nb_outliers2_in_selected_blocks_all.pickle'
with open(filename, 'rb') as f:
	nb_outliers2_in_selected_blocks_all= pickle.load(f)

filename = 'nb_outliers2_in_best_estimator_subsample_all.pickle'
with open(filename, 'rb') as f:
	nb_outliers2_in_best_estimator_subsample_all = pickle.load(f)

filename = 'nb_subsamples_with_no_outlier_all.pickle'
with open(filename, 'rb') as f:
	nb_subsamples_with_no_outlier_all = pickle.load(f)

filename = 'lowest_error_among_computed_estimators_all.pickle'
with open(filename, 'rb') as f:
	lowest_error_among_computed_estimators_all = pickle.load(f)

filename = 'lowest_error_among_basic_estimators_all.pickle'	
with open(filename, 'rb') as f:
	lowest_error_among_basic_estimators_all = pickle.load(f)




####PLOT

import numpy as np

import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
fig_size = [6,4]
plt.rcParams["figure.figsize"] = fig_size
xticks_range = range(0,max(outliers_plus_heavytail_range)+1,16)


# In[ ]:


nswoo = np.array(nb_subsamples_with_no_outlier_all)
plt.plot(outliers_plus_heavytail_range, nswoo.mean(axis=0))
plt.fill_between(outliers_plus_heavytail_range,np.percentile(nswoo, 2.5, axis=0),np.percentile(nswoo, 97.5, axis=0), alpha=.25)
plt.xticks(xticks_range)
plt.xlabel('Number of outliers')
plt.ylabel('Number of subsamples with no outlier')
plt.savefig("number_of_subsamples_without_outliers.png", dpi=150)

plt.clf()
# In[ ]:


no1isb = np.array(nb_outliers1_in_selected_blocks_all)
no1ibes = np.array(nb_outliers1_in_best_estimator_subsample_all)
no2isb = np.array(nb_outliers2_in_selected_blocks_all)
no2ibes = np.array(nb_outliers2_in_best_estimator_subsample_all)

plt2.stackplot(outliers_plus_heavytail_range, no1isb.mean(axis=0),no2isb.mean(axis=0))
plt2.xticks(xticks_range)
plt2.xlabel('Number of outliers')
plt2.ylabel('Number of outliers in subsample of selected estimator')
plt2.legend(['Hard outliers', 'Heavy tails'], numpoints = 1, loc='upper left')
plt2.savefig("number_of_outliers_in_subsample_of_selected_estimator.png", dpi=150)

plt2.clf()
# In[ ]:


no1ibes = np.array(nb_outliers1_in_best_estimator_subsample_all)
no2ibes = np.array(nb_outliers2_in_best_estimator_subsample_all)

plt2.stackplot(outliers_plus_heavytail_range, no1ibes.mean(axis=0),no2ibes.mean(axis=0))
plt2.xticks(xticks_range)
plt2.xlabel('Number of outliers')
plt2.ylabel('Number of outliers in subsample of best estimator')
plt2.legend(['Hard outliers', 'Heavy tails'], numpoints = 1, loc='upper left')
plt2.savefig("number_of_outliers_in_subsample_of_best_estimator.png", dpi=150)

plt2.clf()
# In[ ]:


see = np.array(selected_estimators_errors_all)
leace = np.array(lowest_error_among_computed_estimators_all)
leabe = np.array(lowest_error_among_basic_estimators_all)

plt.fill_between(outliers_plus_heavytail_range, np.log10(np.percentile(see, 2.5, axis=0)),np.log10(np.percentile(see, 97.5, axis=0)), alpha=.25, color='c')

plt.fill_between(outliers_plus_heavytail_range, np.log10(np.percentile(leace, 2.5, axis=0)),np.log10(np.percentile(leace, 97.5, axis=0)), alpha=.25, color='k')

plt.fill_between(outliers_plus_heavytail_range,np.log10(np.percentile(leabe, 2.5, axis=0)), np.log10(np.percentile(leabe, 97.5, axis=0)), alpha=.25)

plt.plot(outliers_plus_heavytail_range, np.log10(see.mean(axis=0)), 'c',
         outliers_plus_heavytail_range, np.log10(leace.mean(axis=0)), 'k',
         outliers_plus_heavytail_range, np.log10(leabe.mean(axis=0))
)

plt.xticks(xticks_range)
plt.xlabel('Number of outliers')
plt.ylabel('log10(error)')
plt.legend(('Selected estimator', 'Best estimator', 'Best basic estimator'), numpoints = 1, loc='center right')
plt.savefig("accuracy.png", dpi=150)
plt.clf()

