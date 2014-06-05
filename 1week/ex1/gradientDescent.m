function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    iter;
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    HX = X * theta;
    theta;
    index_total = length(theta);
    m_total = length(y);
    for index = 1:length(theta)
        sum = 0;
        for m = 1:length(y)
            hx_i = HX(m);
            y_i = y(m);
            x_m = X(m, index);
            sum = sum + (hx_i - y_i) * x_m;
        endfor
        update = sum * alpha/m_total;
        theta(index) = theta(index) - update;
    endfor

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    J = J_history(iter) ;
end
end
