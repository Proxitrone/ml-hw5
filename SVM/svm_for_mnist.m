function [error] = svm_for_mnist(trainX, trainY, testX, testY)
%SVM_FOR_MNIST Summary of this function goes here
%   Detailed explanation goes here

    %% Iniital Training and comparison
    %
    % linear kernel
    model_lin = svmtrain(trainY, trainX, [ '-s 0 -t 0 -b 1 -q']);

    % polynomial kernel
    model_pol = svmtrain(trainY, trainX, [ '-s 0 -t 1 -b 1 -q']);

    % RBF kernel
    model_rbf = svmtrain(trainY, trainX, [ '-s 0 -t 2 -b 1 -q']);
    
    [predicted_label_lin, accuracy_lin, decision_values_lin] = svmpredict(testY, testX, model_lin, [ '-b 1']);

    % polynomial
    [predicted_label_pol, accuracy_pol, decision_values_pol] = svmpredict(testY, testX, model_pol, [ '-b 1']);

    % RBF
    [predicted_label_rbf, accuracy_rbf, decision_values_rbf] = svmpredict(testY, testX, model_rbf, [ '-b 1']);
    
    %% Do a grid search for optimal C (and gamma for RBF)
    N = linspace(0.1, 1.5, 10);
    gamma = linspace(0,2, 10);
    accuracy_rbf = zeros(size(N, 2), size(gamma, 2));
    degree = 1:5;
    
    accuracy_pol = zeros(size(N, 2), size(gamma, 2));
    for n=1:numel(N)
        parfor g=1:numel(gamma)
            %create RBF model
            model = svmtrain(trainY, trainX, [ '-s 0 -t 2 -q -c ', num2str(N(n)), ' -g ', num2str(gamma(g))]);
            [lbl, acc, dec] = svmpredict(testY, testX, model);
            accuracy_rbf(n, g) = acc(1);
        end
        parfor d=1:numel(degree)
            %create Polynomial model
            model = svmtrain(trainY, trainX, [ '-s 0 -t 2 -q -c ', num2str(N(n)), ' -d ', num2str(degree(d))]);
            [lbl, acc, dec] = svmpredict(testY, testX, model);
            accuracy_pol(n, d) = acc(1);
        end
    end
    
    % Plot accuracy matrix for RBF
    figure;
    mesh(N, gamma, accuracy_rbf);
    xlabel('C');
    ylabel('Gamma');
    zlabel('Accuracy');
    title('Gridsearch accuracy RBF');
    
    % Plot accuracy matrix for ploy
    figure;
    mesh(N, degree, accuracy_pol);
    xlabel('C');
    ylabel('Degree');
    zlabel('Accuracy');
    title('Gridsearch accuracy Polynomial');
    
    % Get the optimal parameters from accuracy_rbf
    [~, index] = max(accuracy_rbf, [], 'all','linear');
    [opt_N_rbf, opt_gamma] = ind2sub(size(accuracy_rbf), index);
    opt_gamma = gamma(opt_gamma);
    opt_C_rbf = N(opt_N_rbf);
    
    % Get the optimal parameters from accuracy_pol
    [~, index] = max(accuracy_pol, [], 'all','linear');
    [opt_N_pol, opt_degree] = ind2sub(size(accuracy_pol), index);
    opt_degree = degree(opt_degree);
    opt_C_pol = N(opt_N_pol);
    %% Test optimal on test set
    
    % linear kernel
    model_lin = svmtrain(trainY, trainX, [ '-s 0 -t 0 -b 1 -q -c ', num2str(opt_C_rbf)]);

    % polynomial kernel
    model_pol = svmtrain(trainY, trainX, [ '-s 0 -t 1 -b 1 -q -c ', num2str(opt_C_pol), ' -d ', num2str(opt_degree)]);

    % RBF kernel
    model_rbf = svmtrain(trainY, trainX, [ '-s 0 -t 2 -b 1 -q -c ', num2str(opt_C_rbf), ' -g ', num2str(opt_gamma)]);
    
    % linear
    [predicted_label_lin, accuracy_lin, decision_values_lin] = svmpredict(testY, testX, model_lin, [ '-b 1']);

    % polynomial
    [predicted_label_pol, accuracy_pol, decision_values_pol] = svmpredict(testY, testX, model_pol, [ '-b 1']);

    % RBF
    [predicted_label_rbf, accuracy_rbf, decision_values_rbf] = svmpredict(testY, testX, model_rbf, [ '-b 1']);
    
    %% Create our own kernel: Linear+RBF
    opt_gamma = 1/784;
    numTrain = size(trainX,1);
    numTest = size(testX,1);
    ourKernel_dist = @(X, Y) exp(-opt_gamma.*pdist2(X,Y, 'euclidean'))+X*Y';
    trainOur =  [(1:numTrain)', ourKernel_dist(trainX,trainX)];
    testOur = [(1:numTest)', ourKernel_dist(testX,trainX)];
    % Train our model
    our_model = svmtrain(trainY, trainOur, ' -t 4 -q');
    [lbl, acc, dec] = svmpredict(testY, testOur, our_model);
end

