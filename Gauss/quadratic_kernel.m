function [k] = quadratic_kernel(x_n, x_m, hyper)
%QUADRATIC_KERNEL Exponential of the quadratice form
%   Kernel widely used in for Gaussian process regression
    theta0 = hyper(1);
    theta1 = hyper(2);
    theta2 = hyper(3);
    theta3 = hyper(4);
    
    k = theta0*exp(-theta1/2 * norm(x_n-x_m)^2) + theta2 + theta3*(x_n'*x_m);
end

