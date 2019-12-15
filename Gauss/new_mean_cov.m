function [new_mean, new_cov] = new_mean_cov(x_new, X_train, Y, beta, hyper, train_cov)
%NEW_MEAN_COV Compute the mean and covariance for new points
%   Detailed explanation goes here
    N = size(X_train, 1);
    M = size(x_new, 2);
    k = zeros(N, M);
    c = zeros(M, 1);
    
    for m=1:M
        %Compute vector k as 1 column for each new data point
        for n=1:N
            k(n, m) = quadratic_kernel(X_train(n, 1), x_new(1, m), hyper);
        end
        c(m, 1) = quadratic_kernel(x_new(1, m), x_new(1, m), hyper) + 1/beta;
    end
    %Compute new mean, matrix notation allows to do it quickly for all
    %points
    new_mean = k'/(train_cov)*Y;
    % Compute new variance
    new_cov = zeros(size(new_mean));
    k_transp = k';
    for m=1:M
        new_cov(m, 1) = c(m, 1) - k_transp(m, :)/(train_cov)*k(:, m);
    end
end

