function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


%X m x 2 and theta 2 x 1

HX = X * theta;

J_mext = HX - y;
J_mext  = J_mext';
for item = J_mext
	J = J + item * item;
endfor

J  = J / ( 2 * m);
% =========================================================================

end
