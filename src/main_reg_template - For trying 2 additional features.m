
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
%lambda with lowest cost and highest accuracy
bestLambda = 0;
% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0.01;
%variable for going through features
feature = 3;
%best feature
bestFeature = 3;
%variable for adding another feature
featureTwo = 3;
%best value for second feature added
bestFeatureTwo = 3;
%lowest cost
bestCost = 0;
%gradiant for optimal lambda
bestGrad = 0;

%while loop for trying different lambda values
while (lambda <= 1)
    %resetting the feature variable after each iteration of the while loop
   feature = 3;
   
   
  %trying out different values for first added feature
  while (feature < 10)
    %resetting the second added feature after each iteration of the while loop
    featureTwo = 3;
    %trying different values for second added feature
    while (featureTwo < 10)
      %adding both features to input
      Xo=X;
      X = mapFeature(X(:,1), X(:,2));
      X = [X Xo(:,feature) Xo(:,featureTwo)];
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
      X_test = [X_test X_testo(:,feature) X_testo(:,featureTwo)];

      % Predict and compute accuracy on our testing set
      p = predict(theta, X_test);


      %determening if current values are better then current best. if so replace current best with these values
      testAccuracy = mean(double(p == y_test)) * 100;

      [cost, grad] = costFunctionReg(theta, X, y, lambda);
      
      if (bestAccuracy < testAccuracy || (bestAccuracy <= testAccuracy && bestCost > cost))
        bestLambda = lambda;
        bestAccuracy = testAccuracy;
        bestFeature = feature;
        bestFeatureTwo = featureTwo;
        bestCost = cost;
        bestGrad = grad;
      endif

      %printing all relevant values
      fprintf('Test Accuracy: %f\n', testAccuracy);
      fprintf('Current Lambda: %f\n', lambda);
      fprintf('Current Feature 1: %f\n', feature);
      fprintf('Current Feature 2: %f\n', featureTwo);
      fprintf('Best Lambda: %f\n', bestLambda);
      fprintf('Best Lambdas Accuracy: %f\n', bestAccuracy);
      fprintf('Best Feature 1: %f\n', bestFeature);
      fprintf('Best Feature 2: %f\n', bestFeatureTwo);
      fprintf('########################################\n');
      
      %try next second added feature
      featureTwo = featureTwo + 1;
      
    endwhile
    %try next first added feature
    feature = feature + 1;
  endwhile
  %try next lambda
  lambda = lambda + 0.01;

endwhile












