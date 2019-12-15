function [error] = plot_gpr(x, mu, std, X, Y, hyper)
%PLOT_GPR Plot the Gaussian Process Regression
%   Detailed explanation goes here
    error = 1;
    figure;
    hold on;
    % Fill area between two std curves
    fill([x, fliplr(x)], [(mu+2*std)', fliplr((mu-2*std)')], 'r');
    % Make the area transparent
    alpha(0.25);
    % Plot original trainig data points
    scatter(X, Y, 'Marker','o','MarkerFaceColor','m');
    % Plot mean curve
    plot(x, mu, 'b');
    % Plot upper std curve
    plot(x, mu+2*std, 'r');
    % Plot lower std curve
    plot(x, mu-2*std, 'r');
    hold off;
    % Put hyperparameters in the title
    title(['Gaussian Process Regression (', num2str(hyper(1)), ', ', num2str(hyper(2)), ', ', num2str(hyper(3)), ', ', num2str(hyper(4)), ')']);
    legend('Confidence interval', 'Training data', 'f Mean');
    xlabel('X');
    ylabel('Y');
    
    error = 0;
end

