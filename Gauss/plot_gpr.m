function [error] = plot_gpr(x, mu, std, X, Y, hyper)
%PLOT_GPR Plot the Gaussian Process Regression
%   Detailed explanation goes here
    error = 1;
    
    figure;
    hold on;
    fill([x, fliplr(x)], [(mu+2*std)', fliplr((mu-2*std)')], 'r');
    alpha(0.25);
    scatter(X, Y, 'Marker','o','MarkerFaceColor','m');
    plot(x, mu, 'b');
    plot(x, mu+2*std, 'r');
    plot(x, mu-2*std, 'r');
    hold off;
    title(['Gaussian Process Regression (', num2str(hyper(1)), ', ', num2str(hyper(2)), ', ', num2str(hyper(3)), ', ', num2str(hyper(4)), ')']);
    legend('Confidence interval', 'Training data', 'f Mean');
    xlabel('X');
    ylabel('Y');
    
    error = 0;
end

