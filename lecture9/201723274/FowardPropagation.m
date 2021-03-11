function [J] = FowardPropagation(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed
%
% Part 2: Implement regularization with the cost function
%
a1 = [ones(m, 1), X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2 * Theta2';
h = sigmoid(z3);

y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J = (-1/m)*sum(sum(y.*log(h) + (1-y).*log(1-h)));



% Regularization Part

regTheta1 = Theta1(:, 2:end);
regTheta2 = Theta2(:, 2:end);
error = (lambda/(2*m)) * (sum(sum(regTheta1.^2))+sum(sum(regTheta2.^2)));

J = J+error;

% =============================================================

end

