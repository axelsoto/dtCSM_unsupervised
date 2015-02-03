% Load data
load('miniMNIST');

%Dimensionality of the target space
lDim = 2;

%Parameters for dtCSM
iter = 200;
hDimDist = 'gaussian';
%hDimDist = 'cosine';

batchSize = 1000;

%1 if real data is used, 0 if binary
realData = 0;

%Reduce to lDim dimensions
[mappedData,mapping] = dtCSM_unsup(data, [500,250,50,2], lDim, iter, hDimDist, realData, batchSize);

%Show scatter plot
figure('Name','Unsupervised mapping of 6000 digits from MNIST using dtCSM');
gscatter(mappedData(:,1),mappedData(:,2),labels,[],'.ox+*sdv^<>ph');
save('mappedData','mappedData','labels');


