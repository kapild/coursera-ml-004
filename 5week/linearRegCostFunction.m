function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X(100, n), theta (n,1)


X_theta = X * theta;

X_theta_y  = X_theta - y;
sum_of_sq = X_theta_y' * X_theta_y;

theta_minus_first = theta(2:end); %n-1, 1

sum_of_sq_theta = theta_minus_first' * theta_minus_first;


grad1 = sum_of_sq/(2*m) ;
grad2 = sum_of_sq_theta * lambda/(m* 2);

J  = grad1 + grad2;

X_theta_y_100_1 = X_theta_y;
% =========================================================================

 grad_all_2_1 = X' * X_theta_y_100_1;

 grad_top_one = grad_all_2_1(1:1);

 grad_rest = grad_all_2_1(2:end) ;

 theta_rest = theta(2:end);

theta_j0 = grad_top_one/m;
theta_j_rest =  grad_rest/m + (lambda/m) .* theta_rest;

grad = [theta_j0; theta_j_rest];
end
