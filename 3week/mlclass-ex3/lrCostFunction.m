function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



X_theta = X * theta;

H_theta_X = sigmoid(X_theta);

log_h_theta = log(H_theta_X);
left = y' * log_h_theta;

ones_m = ones(m,1);

ones_minus_h_theta  =  ones_m - H_theta_X;
log_1_minus = log(ones_m - H_theta_X);
ones_minus_y = ones_m - y;

right = ones_minus_y' * log_1_minus;


grad = zeros(size(theta));

H_theta_minus_y = H_theta_X  - y;

grad = (X' * H_theta_minus_y)/m;
regul = (lambda / (2 * m)) *  sum(theta(2:end).^2);

J = -(left + right)/m + regul;

grad_2 = grad(2:end);
grad_2_reg = grad_2 + (lambda)/m * theta(2:end); 
grad = [grad(1:1), grad_2_reg']';
end
