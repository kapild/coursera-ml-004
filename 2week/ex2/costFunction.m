function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


 h_theta_sig = sigmoid(X * theta);
 log_h_theta_sig = log(h_theta_sig);
 left = y' * log_h_theta_sig;
 ones_y = ones(m,1) - y;
 right = ones_y' * log(ones_y - h_theta_sig);

 J = -(left + right)/m;

 
 % X[100,3] * theta[3,1]
 %y[1 * 100] * [100 *1];


 X_theta_100_1 = X * theta;

 sig_X_theta_100_1 = sigmoid(X_theta_100_1);

 sig_X_theta_y_100_1 = sig_X_theta_100_1 - y;

 theta_m = X' * sig_X_theta_y_100_1;

 grad = theta_m/m;
 	
