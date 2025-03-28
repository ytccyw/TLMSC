clear
addpath('./datasets');addpath('./para');addpath('./CM');addpath('./function');
%%%%%%%%%%%%%%%%%%% Load Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%
%% The main parameters, which can be tuned independently
% DIM_PCA=200;%200 or 30
% kpoint=6;%[6 12 18 24 30], Each sample point draws 'kpoint' nearest neighbors as positive examples of triplet
%% Minor parameters, can be ignored for simplicity
% n_random=0;%0 or 1, use or not use hard negative example
% numflag=0;%1 is spectral clustering, 0 is kmeans

datasetName = 'ORL_mtv';
load(['./datasets/',datasetName]);%dataset
load(['./para/',datasetName, '-para.mat']);%Load parameters
N = size(X{1},1);
cls_num = length(unique(Y));
%%%%%%%%%%%%%%%%%%%%% run TLMSC %%%%%%%%%%%%%%%%%%%%%%%%
for v=1:length(X)%use PCA
    dim=size(X{v},2);
    if dim > DIM_PCA
        [X{v},~] = pca(X{v}','NumComponents',DIM_PCA);
        %     X{v} = py.sklearn.decomposition.TruncatedSVD(pyargs('n_components',int64(DIM_PCA),'random_state',int64(0))).fit_transform(X{v});
        %     X{v}=double(X{v});
    end
    X{v}=normalize(X{v},1);
end
for repeat=1:10
    tic;
    em=transformone(X,kpoint,cls_num,n_random);%Obtain improved low-dimensional embeddings
    label = fastSpectralClustering(em,cls_num,numflag);
    TLMSC_res(repeat,:)=Clustering8Measure(Y,label);
    times(repeat)=toc;
end
TLMSC_res=mean(TLMSC_res);
fprintf('TLMSC result: ACC %f   NMI %f \n',TLMSC_res(1)*100,TLMSC_res(2)*100)
mean(times)
