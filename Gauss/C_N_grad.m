function [C_grad] = C_N_grad(C_N, hyper, i, X)
%C_N_GRAD Compute the gradient of training covariance matrix of out
%Gaussian Process Regression
%   Analytically derive the gradients
    C_grad = zeros(size(C_N));
    N = size(C_N, 1);
    if i == 1
        for n=1:N
           for m=1:N
              C_grad(n, m) = exp( -hyper(2)/2* norm(X(n) - X(m))^2);
           end
        end
    elseif i == 2
        for n=1:N
           for m=1:N
              C_grad(n, m) =hyper(1)* exp( -hyper(2)/2* norm(X(n) - X(m))^2) * (- norm(X(n) - X(m))^2)/2;
           end
        end
    elseif i == 3
        C_grad = ones(size(C_N));
    elseif i == 4 
        for n=1:N
           for m=1:N
              C_grad(n, m) =  X(n)'*X(m);
           end
        end
    end
end

