function [t, y, tr] = neuralnet(algo)

load input_data.mat

% Inputs and Targets
x = test_input;
t = test_targets;

% Training Algorithm
trainFcn = algo;
% Set Hidden Layer Size
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

net.trainParam.max_fail = 1000;
net.trainParam.showWindow = false;
%net.trainParam.showCommandLine = false;

% Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'divideind';
[trainInd, valInd, testInd] = divideind(463,1:369,370:417,418:463);
net.divideParam.trainInd = trainInd;
net.divideParam.testInd  = testInd;
net.divideParam.valInd   = valInd;

% Choose a Performance Function
net.performFcn = 'mse';  

disp('Starting Neural Network Training');
% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

% Save the Network
trained_net_admin = net;
save trained_net_admin;

disp('Neural Network Training Completed');

end
