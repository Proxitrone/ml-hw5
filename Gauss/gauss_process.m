function [mu_f, std_f] = gauss_process(data_mat, beta, kernel_hyper_param)
%GAUSS_PROCESS Gaussian Process Regression
%   We want to predict distribution of f given X and Y, where Y = f(X)+eps
    
    X = data_mat(:, 1);
    Y = data_mat(:, 2);
    N = size(X, 1);
    %% Initial set of hyperparameters for our kernel
%     kernel_hyper_param = [1, 1, 1, 1]; 
    delta = 1;
    %% Computing the covariance C_N for training
    C_N = gauss_covariance(X, N, beta, delta, kernel_hyper_param);
    
    %% Generate new points X and compute mean and std of f
    x_new = linspace(-60, 60);
    [mu_f, std_f] = new_mean_cov(x_new, X, Y, beta, kernel_hyper_param, C_N);
    
    %% Plot gaussian process regression with initial hyperparameters
    plot_gpr(x_new, mu_f, std_f, X, Y, kernel_hyper_param);
    
    %% Optimize hyperparameters
    % Our objective is the log-likelihood,  we need to take the gradient of
    % it w.r.t each hyperparameter theta and make a gradient ascent step
    % into that direction
    opt_max_iter = 200;
    hyper_num = size(kernel_hyper_param, 2);
    % Objectie is non-convex, so the starting point really matters in our
    % gradient based optimization
%     kernel_hyper_param = [1, 1, 1, 1];
%     C_N = gauss_covariance(X, N, beta, delta, kernel_hyper_param);
    objective = [];
    % Objective: -1/2 * ln(abs(C_N) - 1/2 *Y'\C_N*Y - N/2*ln(2*pi))
    objective = [objective, log_likelihood(C_N, Y, N)];
    obj_grad = zeros(hyper_num, 1);
    alpha = ([0.01; 0.01; 0.01; 0.01]);
    alpha = (alpha./(1+log(1:opt_max_iter)));
%     alpha = 0.02 * ones(1, opt_max_iter);
    epsilon = 1e-3;
    for k=1:opt_max_iter
        % We have 4 hyperparameters in our kernel function
        % In order to take the gradient of the objective, we need to take
        % the derivative of C_N w.r.t each of the hyperparameters
        for i=1:hyper_num
            C_grad_i = C_N_grad(C_N, kernel_hyper_param, i, X);
            obj_grad(i, 1) = log_likelihood_grad_i(C_N, C_grad_i, Y);
        end
        % Make an ascent step in the direction of the gradient
        % Need a stepsize sequence alpha
        step = alpha(:, k).*obj_grad;
        kernel_hyper_param = kernel_hyper_param + step';
        % Recompute covariance for training and objective
        C_N = gauss_covariance(X, N, beta, delta, kernel_hyper_param);
        objective = [objective, log_likelihood(C_N, Y, N)];
        % Check the termination criteria (Increase in the objective<epsilon)
        if abs(objective(end)-objective(end-1))<epsilon
            disp(['Termination reached']);
            disp(['Initial objective: ', num2str(objective(1)), ' Final objective: ', num2str(objective(end))]);
            break;
        end
    end
    %% Plot graph for new sub-optimal hyperparameters
    [mu_f, std_f] = new_mean_cov(x_new, X, Y, beta, kernel_hyper_param, C_N);
    plot_gpr(x_new, mu_f, std_f, X, Y, kernel_hyper_param);
end

