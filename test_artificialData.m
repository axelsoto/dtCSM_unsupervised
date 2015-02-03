%Dimensionality of the raw instances (should be an even number for this
%example)
hDim = 50;

%Number of instances per class
instPerClass = 100;

%Dimensionality of the target space
lDim = 2;

%Parameters for dtCSM
iter = 5;
%hDimDist = 'gaussian';
hDimDist = 'cosine';
batchSize = fix(instPerClass/3);

%1 if real data is used, 0 if binary
realData = 1;

%Set means of the three classes
mu1 = repmat([-5],1,hDim);
mu2 = repmat([5,-5],1,fix(hDim/2));
mu3 = repmat([5],1,hDim);

%generate random samples
mu = [repmat(mu1,instPerClass ,1);repmat(mu2,instPerClass ,1);repmat(mu3,instPerClass ,1)];
sigma = eye(hDim);

rPerm = randperm(instPerClass*3);

data = mvnrnd(mu(rPerm,:),sigma);

%Reduce to lDim dimensions
[mappedData,mapping] = dtCSM_unsup(data, [fix(hDim/2),2], lDim, iter, hDimDist, realData, batchSize);

%Show scatter plot
figure('Name','Unsupervised mapping using dtCSM. Artificially-generated data');
colors = [repmat([1],instPerClass ,1);repmat([2],instPerClass ,1);repmat([3],instPerClass ,1)];
scatter(mappedData(:,1),mappedData(:,2),3,colors(rPerm,:));


