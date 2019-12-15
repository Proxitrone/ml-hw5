load input.data
beta = 5;
kernel_hyper_param = [1, 1, 1, 1];
gauss_process(input, beta, kernel_hyper_param);

load X_test.csv
load Y_test.csv
load X_train.csv
load Y_train.csv

svm_for_mnist(X_train, Y_train, X_test, Y_test);

