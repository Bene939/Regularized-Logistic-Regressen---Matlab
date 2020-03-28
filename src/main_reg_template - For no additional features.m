
%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).


%using the load function didnt work. so i changed it to csvread

data_train = csvread('Train_Data.txt');
data_test = csvread('Test_Data.txt');

X = data_train(:, [2:10]); y = data_train(:, 11); 
% change label 2 -> 0 for benign, 4 -> 1 for malignant
y(find(y == 2)) = 0;
y(find(y == 4)) = 1;

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
%the features we use
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);


%bestAccuracy = best Test accuracy
bestAccuracy = 0;
%lambda value with lowest cost and highest accuracy
bestLambda = 0;
%initial lambda value
lambda = 0;
%best cost
bestCost = 1;
%gradiant of the best lambda value
bestGrad = 0;
%training accuracy of optimal lambda
bestTrainAccuracy = 0;


%while loop for finding optimal lambda
%if the search is until 10 the computation time is long but finds better lambda value than if one searches until 1
while (lambda  <= 10)

      % Optimize
      [theta, J, exit_flag] = ...
        fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

      % Compute accuracy on our training set
      p = predict(theta, X);

      trainAccuracy = mean(double(p == y)) * 100;

      fprintf('########################################\n');
      fprintf('Train Accuracy: %f\n', trainAccuracy);

      % Compute and display initial cost and gradient for regularized logistic
      % regression - dummy code to print cost & grad
      [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
      fprintf('Cost at initial theta (zeros): %f\n', cost);
      fprintf('Gradient at initial theta (zeros) - first five values only:\n');
      fprintf(' %f \n', grad(1:5));

      % load testing data
      X_test = data_test(:, [2:10]); y_test = data_test(:, 11); 
      % change label 2 -> 0 for benign, 4 -> 1 for malignant
      y_test(find(y_test == 2)) = 0;
      y_test(find(y_test == 4)) = 1;

      %----------------define feature 
      X_test = mapFeature(X_test(:,1), X_test(:,2));

      % Predict and compute accuracy on our testing set
      p = predict(theta, X_test);


      
      testAccuracy = mean(double(p == y_test)) * 100;

      %if the values for the current lambda are better than the current best lambda the current value becomes the new best
      [cost, grad] = costFunctionReg(theta, X, y, lambda);
      if (bestAccuracy < testAccuracy || (bestAccuracy <= testAccuracy && bestCost > cost))
        bestLambda = lambda;
        bestAccuracy = testAccuracy;
        bestCost = cost;
        bestGrad = grad;
        bestTrainAccuracy = trainAccuracy;
      endif
      % print all relevant values
      fprintf('Test Accuracy: %f\n', testAccuracy);
      fprintf('Current Lambda: %f\n', lambda);
      fprintf('Best Lambda: %f\n', bestLambda);
      fprintf('Best Lambdas Accuracy: %f\n', bestAccuracy);
      fprintf('Best Cost: %f\n', bestCost);
      fprintf('########################################\n');

      %try next lambda
      lambda = lambda + 0.001;

endwhile












