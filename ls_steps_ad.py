import numpy as np 
import pandas as pd 
import copy

import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#read in the training set
df = pd.read_csv("~/Desktop/least_squares/data_ad/Advertising.csv")
df.to_csv('orig_data.csv')

#set the train/test split size
train_size = .7
test_size = 1-train_size

#pick the feature(s)
feature_labels = ['TV', 'Radio', 'Newspaper']
#feature_labels = ['Radio']
target_label = ['Sales']

features = df[feature_labels]
target = df[target_label]

features_np = features.to_numpy()
target_np = target.to_numpy()

#now split into train/test
total_samples = target_np.shape[0]
train_samples = int(train_size*total_samples)
test_samples = total_samples - train_samples

features_train = features_np[:train_samples]
target_train = target_np[:train_samples]

features_test = features_np[train_samples:]
target_test = target_np[train_samples:]

#run the alg w/ the normal equations
X = features_train
y = target_train
theta = (np.linalg.inv((X.T)@X))@((X.T)@y)

X_test = features_test
y_test = target_test
y_preds_train = (X@theta)
y_preds_test = (X_test@theta)

train_error = mean_squared_error(y, y_preds_train)
test_error = mean_squared_error(y_test, y_preds_test)

print ('Norm Theta')
print (theta)

print ('Norm Train Error')
print (train_error)

print ('Norm Test Error')
print (test_error)


#plot this
# plt.figure(1)
# plt.scatter(X[:, 0], y[:, 0], color='blue')
# plt.plot(X[:, 0], y_preds_train[:, 0], color='red', label='LR Prediction')
# plt.legend()
# plt.xlabel('Radio Budget')
# plt.ylabel('Sales')
# plt.show()

#now we run the algorithm w/ sklearn
lr = LinearRegression(fit_intercept=False)
lr.fit(features_train, target_train)

lr_theta = lr.coef_

target_train_preds = lr.predict(features_train)
target_test_preds = lr.predict(features_test)

train_error = mean_squared_error(target_train, target_train_preds)
test_error = mean_squared_error(target_test, target_test_preds)

print ('LR Theta')
print (lr_theta)

print ('Train Error')
print (train_error)

print ('Test Error')
print (test_error)


#add random feature, make plot
num_random_features = 100

features_rand = copy.deepcopy(features_np)

train_errors = np.zeros((num_random_features))
test_errors = np.zeros((num_random_features))

for i in range(num_random_features):
	rand = np.random.normal(0, 1, total_samples).reshape(total_samples, 1)
	features_rand = np.concatenate((features_rand, rand), axis=1)

	features_train = features_rand[:train_samples]
	features_test = features_rand[train_samples:]

	lr = LinearRegression(fit_intercept=False)
	lr.fit(features_train, target_train)

	target_train_preds = lr.predict(features_train)
	target_test_preds = lr.predict(features_test)

	train_error = mean_squared_error(target_train, target_train_preds)
	test_error = mean_squared_error(target_test, target_test_preds)

	train_errors[i] = train_error
	test_errors[i] = test_error


#plot this
x = np.linspace(0, num_random_features, num=num_random_features)

plt.figure(2)
plt.plot(x, train_errors, color='red', label='train error')
plt.plot(x, test_errors, color='blue', label='test error')
plt.legend()
plt.xlabel('# Random Features')
plt.ylabel('Mean Squared Error')
#plt.show()

#add engineered feature
features_rand = copy.deepcopy(features_np)
features_eng = copy.deepcopy(features_np)

num_eng_features = 14
eng_noise = .001
eng_features = np.zeros((total_samples, num_eng_features))

eng_features[:, 0] = features_np[:, 0]*features_np[:, 0] 
eng_features[:, 1] = features_np[:, 1]*features_np[:, 1] 
eng_features[:, 2] = features_np[:, 2]*features_np[:, 2] 

eng_features[:, 3] = features_np[:, 0]*features_np[:, 1] 
eng_features[:, 4] = features_np[:, 0]*features_np[:, 2] 
eng_features[:, 5] = features_np[:, 1]*features_np[:, 2] 

eng_features[:, 6] = (features_np[:, 0]*features_np[:, 1]*features_np[:, 2])

eng_features[:, 7] = features_np[:, 0]*features_np[:, 0]*features_np[:, 0]
eng_features[:, 8] = features_np[:, 1]*features_np[:, 1]*features_np[:, 1]
eng_features[:, 9] = features_np[:, 2]*features_np[:, 2]*features_np[:, 2]

eng_features[:, 10] = features_np[:, 0]*features_np[:, 1]*features_np[:, 1]
eng_features[:, 11] = features_np[:, 0]*features_np[:, 2]*features_np[:, 2]
eng_features[:, 12] = features_np[:, 1]*features_np[:, 2]*features_np[:, 2]

eng_features[:, 13] = (features_np[:, 0]*features_np[:, 1]*features_np[:, 2])

#some features w/ noise
# half_eng_features = int(num_eng_features/2)
# for i in range(half_eng_features):
# 	eng_features[:, i+half_eng_features] = eng_features[:, i] + np.random.normal(0, eng_noise, total_samples)


train_rand_errors = np.zeros((num_eng_features))
test_rand_errors = np.zeros((num_eng_features))

train_eng_errors = np.zeros((num_eng_features))
test_eng_errors = np.zeros((num_eng_features))

for i in range(num_eng_features):
	rand = np.random.normal(0, 1, total_samples).reshape(total_samples, 1)
	features_rand = np.concatenate((features_rand, rand), axis=1)

	eng = eng_features[:, i].reshape(total_samples, 1)
	features_eng = np.concatenate((features_eng, eng), axis=1)

	features_rand_train = features_rand[:train_samples]
	features_rand_test = features_rand[train_samples:] 

	features_eng_train = features_eng[:train_samples]
	features_eng_test = features_eng[train_samples:]

	#fit and test rand
	lr = LinearRegression(fit_intercept=False)
	lr.fit(features_rand_train, target_train)

	target_train_rand_preds = lr.predict(features_rand_train)
	target_test_rand_preds = lr.predict(features_rand_test)

	train_error = mean_squared_error(target_train, target_train_rand_preds)
	test_error = mean_squared_error(target_test, target_test_rand_preds)

	train_rand_errors[i] = train_error
	test_rand_errors[i] = test_error

	#fit and test eng
	lr = LinearRegression(fit_intercept=False)
	lr.fit(features_eng_train, target_train)

	target_train_eng_preds = lr.predict(features_eng_train)
	target_test_eng_preds = lr.predict(features_eng_test)

	train_error = mean_squared_error(target_train, target_train_eng_preds)
	test_error = mean_squared_error(target_test, target_test_eng_preds)

	train_eng_errors[i] = train_error
	test_eng_errors[i] = test_error

#plot this
x = np.linspace(0, num_eng_features, num=num_eng_features)

plt.figure(3)
plt.plot(x, train_rand_errors, color='red', label='rand train error')
plt.plot(x, test_rand_errors, color='blue', label='rand test error')
plt.plot(x, train_eng_errors, color='cyan', label='eng train error')
plt.plot(x, test_eng_errors, color='green', label='eng test error')
plt.legend()
plt.xlabel('# Features')
plt.ylabel('Mean Squared Error')
#plt.show()

#compare dependent, independent features
features_pure = copy.deepcopy(features_np)
features_eng = copy.deepcopy(features_np)
features_rand = copy.deepcopy(features_np)

#zero out the last feature
features_eng[:, 2] = 0
features_rand[:, 2] = 0

#add engineered feature
eng = eng_features[:, 3]
features_eng[:, 2] = eng

#add random feature
rand = np.random.normal(0, 1, total_samples)
features_rand[:, 2] = rand

features_pure_train = features_pure[:train_samples]
features_pure_test = features_pure[train_samples:]

features_eng_train = features_eng[:train_samples]
features_eng_test = features_eng[train_samples:]

features_rand_train = features_rand[:train_samples]
features_rand_test = features_rand[train_samples:]

#fit pure model
lr = LinearRegression(fit_intercept=False)
lr.fit(features_pure_train, target_train)

target_train_pure_preds = lr.predict(features_pure_train)
target_test_pure_preds = lr.predict(features_pure_test)

train_error_pure = mean_squared_error(target_train, target_train_pure_preds)
test_error_pure = mean_squared_error(target_test, target_test_pure_preds)

#fit engineered model
lr = LinearRegression(fit_intercept=False)
lr.fit(features_eng_train, target_train)

target_train_eng_preds = lr.predict(features_eng_train)
target_test_eng_preds = lr.predict(features_eng_test)

train_error_eng = mean_squared_error(target_train, target_train_eng_preds)
test_error_eng = mean_squared_error(target_test, target_test_eng_preds)

print ('Pure Errors')
print (train_error_pure)
print (test_error_pure)

print ('Eng Errors')
print (train_error_eng)
print (test_error_eng)

#consider cov matrices
X_pure = features_pure_train
X_eng = features_eng_train
X_rand = features_rand_train

cov_pure = np.cov(X_pure.T)/total_samples
cov_eng = np.cov(X_eng.T)/total_samples
cov_rand = np.cov(X_rand.T)/total_samples


#code for different number of training examples
test_train_start = 50
test_train_end = total_samples-50
total_errors = test_train_end-test_train_start

train_errors = np.zeros((total_errors))
test_errors = np.zeros((total_errors))

cnt = 0
for i in range(test_train_start, test_train_end):
	features_train = features_np[:i]
	target_train = target_np[:i]

	features_test = features_np[i:]
	target_test = target_np[i:]

	#fit model
	lr = LinearRegression(fit_intercept=False)
	lr.fit(features_train, target_train)

	train_preds = lr.predict(features_train)
	test_preds = lr.predict(features_test)

	train_error = mean_squared_error(target_train, train_preds)
	test_error = mean_squared_error(target_test, test_preds)

	train_errors[cnt] = train_error
	test_errors[cnt] = test_error

	cnt += 1


#plot this
x = np.linspace(test_train_start, test_train_end, num=total_errors)

plt.figure(4)
plt.plot(x, train_errors, color='red', label='train error')
plt.plot(x, test_errors, color='blue', label='test error')
plt.legend()
plt.xlabel('# Training Examples')
plt.ylabel('Mean Squared Error')
#plt.show()
