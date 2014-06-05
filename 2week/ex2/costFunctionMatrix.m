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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%X is (m, n+1), , initial_theta = n + 1
% Initialize fitting parameters
%initial_theta = zeros(n + 1, 1);

Xtheta = sigmoid(X * theta);

a = y' * log(Xtheta) ;

one_m = ones(m,1);
b = (one_m - y)' * log(1 - Xtheta);
J = -1 * (a+b)/m;


%sum =0;
%for i = 1:m
 %   y_i = y(i);
  %  X_i = X(i,:);
  %  h_o_x_i = X_i * theta;
  %  a = log(sigmoid(h_o_x_i));
  %  b = log(sigmoid(1 - h_o_x_i));
  %  sum = (y_i * a ) + ((1 - y_i) * b);
%endfor


% =============================================================

end
