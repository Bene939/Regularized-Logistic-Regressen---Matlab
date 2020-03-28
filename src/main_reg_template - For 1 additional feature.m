
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


%bestAccuracy = best Test accuracy
bestAccuracy = 0;
%lambda with the lowest cost and highest testing accuracy 
bestLambda = 0;
% initial lambda
lambda = 0;
%initial feature
feature = 3;
%best feature
bestFeature = 3;
%best cost
bestCost = 0;
%gradiant for best lambda
bestGrad = 0;


%while loop for trying different lambda
while (lambda <= 1)
  
   %resetting the feature value after each iteration of the while loop
   feature = 3;
  
  %while loop for trying different features
  while (feature < 10)
    
      %adding one feature to use
      Xo=X;
      X = mapFeature(X(:,1), X(:,2));
      X = [X Xo(:,feature)];
      % Initialize fitting parameters
      initial_theta = zeros(size(X, 2), 1);
      % Set Options
      options = optimset('GradObj', 'on', 'MaxIter', 400);
      


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
      X_testo = X_test;
      X_test = mapFeature(X_test(:,1), X_test(:,2));
      X_test = [X_test X_testo(:,feature)];

      % Predict and compute accuracy on our testing set
      p = predict(theta, X_test);

      
      %detemening if current lambda value gives better result than current best. if so this becomes the new best
      testAccuracy = mean(double(p == y_test)) * 100;

      [cost, grad] = costFunctionReg(theta, X, y, lambda);
      
      if (bestAccuracy < testAccuracy || (bestAccuracy <= testAccuracy && bestCost > cost))
        bestLambda = lambda;
        bestAccuracy = testAccuracy;
        bestFeature = feature;
        bestCost = cost;
        bestGrad = grad;
      endif

      %printing out all relevant info
      fprintf('Test Accuracy: %f\n', testAccuracy);
      fprintf('Current Lambda: %f\n', lambda);
      fprintf('Current Feature 1: %f\n', feature);
      fprintf('Best Lambda: %f\n', bestLambda);
      fprintf('Best Lambdas Accuracy: %f\n', bestAccuracy);
      fprintf('Best Feature 1: %f\n', bestFeature);
      fprintf('########################################\n');
      
      %trying next feature
      feature = feature + 1;
  endwhile
  %trying next lambda
  lambda = lambda + 0.01;

endwhile












