function [C_N] = gauss_covariance(X, N, beta, delta, hyper)
%GAUSS_COVARIANCE Compute the covariance matrix of our training data
%   Use quadratic kernel
    C_N = zeros(N);
    for n=1:N
        for m=1:N
            C_N(n, m) = quadratic_kernel(X(n,1), X(m,1), hyper)+ (1/beta)*delta;
        end
    end
end

