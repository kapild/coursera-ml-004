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




 h_theta_sig = sigmoid(X * theta);
 log_h_theta_sig = log(h_theta_sig);
 left = y' * log_h_theta_sig;


 ones_y = ones(m,1) - y;
 right = ones_y' * log(ones_y - h_theta_sig);

 J1 = -(left + right)/m;

 theta_t= theta';
 theta_reg= theta_t(:,2,end);

 theta_reg = theta_reg';
 sq_J = (theta_reg' * theta_reg);

 J2 = sq_J * lambda/2;
 J2_m  = J2/m;

 J = J2_m + J1;

% =============================================================

end
