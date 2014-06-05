function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

 errorNow = ones(length(steps) * length(steps), 1);
 errorNow = 100 *errorNow;
 for index  = 1 : length(steps)
 	C = steps(index);
 	for k = 1 : length(steps)
 		sigma = steps(k);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		loc = k + length(steps) * (index-1);
		errorNow(loc) = mean(double(predictions ~= yval));
	endfor
endfor
[err , indx ]= min(errorNow);

k = mod(indx , length(steps));
index = floor(indx / length(steps));

C = steps(index);
sigma  = steps(k);

% =========================================================================

end
