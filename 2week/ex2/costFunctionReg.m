function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% X[128,28] * theta[28,1]

 h_theta_sig = sigmoid(X * theta);
 log_h_theta_sig = log(h_theta_sig);

 %y[1 * 128] * [128 *1];
 left = y' * log_h_theta_sig;
 ones_y = ones(m,1) - y;
 right = ones_y' * log(ones_y - h_theta_sig);
 J1 = -(left + right)/m;
 theta_req = theta(2:end);
 J2_temp = theta_req' * theta_req;
 J2 = J2_temp * lambda/(2 * m);
 J = J1 + J2;

% =============================================================

 X_theta_100_1 = X * theta;

 sig_X_theta_100_1 = sigmoid(X_theta_100_1);

 sig_X_theta_y_100_1 = sig_X_theta_100_1 - y;

 theta_m = X' * sig_X_theta_y_100_1;

 theta_m_one  = theta_m(1);
 theta_m_rest = theta_m(2:end) + lambda  .* theta(2:end);

 grad  = [theta_m_one; theta_m_rest]/m;	
% =============================================================

end
