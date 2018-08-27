import numpy as np
from sklearn import datasets, linear_model, metrics

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# regr = linear_model.LinearRegression()
#
# regr.fit(diabetes_X_train, diabetes_y_train)
#
# diabetes_y_pred = regr.predict(diabetes_X_test)
#
# print('Coefficients: \n', regr.coef_)
#
# mean_squared_error = metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred)
# print("Mean squired erro: %.2f" % mean_squared_error)
# print("="*120)

X = diabetes_X_train
y = diabetes_y_train

W = np.random.random(10)
b = 0.001

learning_rate = 0.1
epochs = 100000

for i in range(epochs):
    # prediction
    preds = np.dot(X, W) + b
    # error and cost
    error = preds - y
    mean_squared_error = np.mean(np.power(error, 2))
    # mean_squared_error = 0
    # for s in range(len(preds)):
    #     mean_squared_error += (preds[s] - y[s]) ** 2
    # mean_squared_error /= len(preds)
    # gradients
    gradients_w = error.dot(X) / len(preds)
    gradients_b = error.sum() / len(preds)

    W = W - gradients_w * learning_rate
    b = b - gradients_b * learning_rate

    if i % 5000 == 0:
        print("Epoch %d: %f" % (i, mean_squared_error))

X = diabetes_X_test
y = diabetes_y_test

preds = np.dot(X, W) + b
cost = 0
mean_squared_error = 0
for i in range(len(preds)):
    mean_squared_error += (preds[i] - y[i]) ** 2
mean_squared_error /= len(preds)

print('Coefficients: \n', W)
print("Mean squared error: %.2f" % mean_squared_error)
print("="*120)
