clc;clear;close all
imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

imsize = [227,227,3];
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
augTrain = augmentedImageDatastore(imsize,imdsTrain);
augVal = augmentedImageDatastore(imsize,imdsValidation);
layers = [
    imageInputLayer([227 227 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',5, ...
    'MiniBatchSize',64,...
    'Shuffle','every-epoch', ...
    'GradientThreshold',1,...
    'ValidationData',augVal,...
    'ValidationFrequency',1,...
    'Verbose',true, ...
    'VerboseFrequency',1,...
    'Plots','training-progress');

%%
net = trainNetwork(augTrain,layers,options);
%%
[YPred,scores] = classify(net,augVal);
pred_acc = sum(YPred == imdsValidation.Labels)/length(YPred)
%%
clc;
[file,~] = uigetfile({'*.jpg';'*.jpeg';'*.png'},'Select file');
test_image = imread(file);
figure;imshow(test_image);
[YPred,scores] = classify(net,test_image)
