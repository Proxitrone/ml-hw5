function [C_N] = gauss_covariance(X, N, beta, delta, hyper)
%GAUSS_COVARIANCE Summary of this function goes here
%   Detailed explanation goes here
    C_N = zeros(N);
    for n=1:N
        for m=1:N
            C_N(n, m) = quadratic_kernel(X(n,1), X(m,1), hyper)+ (1/beta)*delta;
        end
    end
end

