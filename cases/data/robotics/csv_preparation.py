import numpy as np
from sklearn.model_selection import train_test_split
import os
from core.utils import project_root

# download dataset from https://www.kaggle.com/ugocupcic/grasping-dataset
full_path_dataset = input("Enter absolute path to grasping-dataset: ")
dataset = np.loadtxt(full_path_dataset, skiprows=1, usecols=range(1, 30), delimiter=",")
with open('shadow_robot_dataset.csv', 'r') as f:
    header = f.readline()
header = header.strip("\n").split(',')
header = [i.strip(" ") for i in header]
saved_cols = []
for index, col in enumerate(header[1:]):
    if ("vel" in col) or ("eff" in col):
        saved_cols.append(index)
new_X = []
for x in dataset:
    new_X.append([x[i] for i in saved_cols])
# new_X.append(new_X[0])
X = np.array(new_X)
Y = np.array(dataset[:, 0]).reshape((len(dataset[:, 0]), 1))
seed = 7
np.random.seed(seed)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)

# this is a sensible grasp threshold for stability
GOOD_GRASP_THRESHOLD = 100

# we're also storing the best and worst grasps of the test set to do some sanity checks on them
itemindex = np.where(Y_test > 1.05*GOOD_GRASP_THRESHOLD)
best_grasps = X_test[itemindex[0]]
itemindex = np.where(Y_test <= 0.95*GOOD_GRASP_THRESHOLD)
bad_grasps = X_test[itemindex[0]]

# discretizing the grasp quality for stable or unstable grasps
Y_train = np.array([int(i > GOOD_GRASP_THRESHOLD) for i in Y_train])
Y_test = np.array([int(i > GOOD_GRASP_THRESHOLD) for i in Y_test])
data_train = np.hstack((np.arange(1, X_train.shape[0]+1).reshape((X_train.shape[0], 1)), X_train, Y_train))
data_test = np.hstack((np.arange(1, X_test.shape[0]+1).reshape((X_test.shape[0], 1)), X_test, Y_test))
file_path_test = 'cases/data/robotics/robotics_data_test.csv'
full_path_test = os.path.join(str(project_root()), file_path_test)
file_path_train = 'cases/data/robotics/robotics_data_train.csv'
full_path_train = os.path.join(str(project_root()), file_path_train)
np.savetxt(full_path_train, data_train, delimiter=",")
np.savetxt(full_path_test, data_test, delimiter=",")
