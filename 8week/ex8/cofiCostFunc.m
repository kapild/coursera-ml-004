function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
[m n] = size(R);
sum_y = 0;
for i = 1: m
    for j = 1 : n
        if(R(i,j) == 1)
            theta_x= Theta(j,:) * X(i,:)';
            y = Y (i,j);
            diff_1= theta_x - y;
            diff_2 = diff_1 * diff_1;
            sum_y+=diff_2;
        endif
    endfor
endfor
        
J1 = sum_y/2;

theta_sq  = Theta .^2;
J2 = sum(sum(theta_sq));

x_sq  = X .^2;
J3 = sum(sum(x_sq));

J = J1 + (J2 + J3) * lambda/2;

size_of_Theta = size(Theta);

size_of_X_grad = size(X_grad);
num_movies = num_movies;
num_features = num_features;
num_users = num_users;
size_of_r = size(R);

for i = 1 : num_movies
    for k = 1 : num_features
        some_val = 0;
        for j= 1 : num_users
            if(R(i,j) == 1)
                theta_x= Theta(j,:) * X(i,:)';
                y_i = Y (i,j);
                diff_1= theta_x - y_i;
                some_val += (diff_1 * Theta(j,k));
            endif
        endfor
        X_grad(i,k) = some_val + lambda * X(i,k);
    endfor
endfor



size_of_Theta_grad = size(Theta_grad);
num_users = num_users;
num_features = num_features;



for j = 1 : num_users
    for k = 1 : num_features
        some_val = 0;
        for i = 1: num_movies
            if(R(i,j) == 1)
                theta_x= Theta(j,:) * X(i,:)';
                y_i = Y (i,j);
                diff_1= theta_x - y_i;
                some_val += (diff_1 * X(i,k));
            endif
        endfor
        Theta_grad(j,k) = some_val + lambda * Theta(j,k);
    endfor
endfor


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
