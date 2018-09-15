
%% Load a deep, convolutional neural network
alex = alexnet;
layers = alex.Layers

%% Modify the network to use prefered number of categories
layers(23) = fullyConnectedLayer(2); 
layers(25) = classificationLayer

%% Load training data as per requirement
allImages = imageDatastore('E:\myImages', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainingImages, testImages] = splitEachLabel(allImages, 0.7, 'randomize');

%% Re-train the Deep Neural Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 20, 'MiniBatchSize', 64);
myNet = trainNetwork(trainingImages, layers, opts);

%% Measure network accuracy after retraining
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

