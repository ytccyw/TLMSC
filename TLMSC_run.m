clear
if isempty(gcp('nocreate'))
    parpool('IdleTimeout',60);
end
addpath('./datasets');addpath('./para');
addpath('./CM');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datasetName = 'ORL_mtv';
load(['./datasets/',datasetName]);
load(['./para/',datasetName]);
% DIM_PCA: 200 or 30
% n_random: 0 or 1, use or not use hard negative example
% neidian: [6 12 18 24 30], Each sample point draws 'neidian' nearest neighbors as positive examples of triplet
N = size(X{1},1);
cls_num = length(unique(Y));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
parfor v=1:length(X)%Obtain initial low-dimensional embedding
    dim=size(X{v},2);
    if dim > DIM_PCA
        X{v} = py.sklearn.decomposition.TruncatedSVD(pyargs('n_components',int64(DIM_PCA),'random_state',int64(0))).fit_transform(X{v});
        X{v}=double(X{v});
    end
    X{v}=normalize(X{v},1);
end
em=transformone(X,neidian,cls_num,n_random);%Obtain improved low-dimensional embeddings
if numflag==0.5
    num=floor(N^0.5);
else
    num=ceil(N/10);
end
label = fastSpectralClustering(em,cls_num,num);
toc
disp("    acc       nmi       F       Precision   AR        PURITY    RECALL")
disp(Clustering8Measure(Y, label))

