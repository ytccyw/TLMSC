clear
addpath('./datasets');addpath('./para');addpath('./function');addpath('./CM');
%%%%%%%%%%%%%%%%%%% Load Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%
datasetName = 'yaleB';%【【dataset name can be changed: yaleB, flower17, Caltech101, COIL20MV】】
load(['./datasets/',datasetName]);
REPEAT=10;%Number of repeats. For Caltech101 can be set smaller
N = size(X{1},1);
XX=X;
cls_num = length(unique(Y));
%DIM_PCA: 200 or 30
%n_random: 0 or 1, use or not use hard negative example
%neidian: [6 12 18 24 30], Each sample point draws 'neidian' nearest neighbors as positive examples of triplet
%%%%%%%%%%%%%%%%%%%%% run TLMSC %%%%%%%%%%%%%%%%%%%%%%%%
load(['./para/',datasetName,'_TLMSC']);%load parameters
for v=1:length(X)%use PCA
    dim=size(X{v},2);
    if dim > DIM_PCA
        X{v} =py.sklearn.decomposition.PCA(pyargs('n_components',int64(DIM_PCA))).fit_transform(X{v});
        X{v} =py.numpy.ascontiguousarray(X{v});
        X{v}=double(X{v});
    end
    X{v}=normalize(X{v},1);
end
for repeat=1:REPEAT
    em=transformone(X,neidian,cls_num,n_random);%Obtain improved low-dimensional embeddings
    label = fastSpectralClustering(em,cls_num,numflag);
    TLMSC_res(repeat,:)=Clustering8Measure(Y,label);
end
TLMSC_res=mean(TLMSC_res);
fprintf('TLMSC result: ACC %f   NMI %f \n',TLMSC_res(1)*100,TLMSC_res(2)*100)
%%%%%%%%%%%%%%%%%%%%% run TLMSC without PCA %%%%%%%%%%%%%%%%%%%%%%%%
load(['./para/',datasetName,'_TLMSC_noPCA']);%load parameters
X=XX;
for v=1:length(X)
    X{v}=normalize(X{v},1);%【If an error is reported, % this】
end
for repeat=1:REPEAT
    em=transformone(X,neidian,cls_num,n_random);%Obtain improved low-dimensional embeddings
    label = fastSpectralClustering(em,cls_num,numflag);
    TLMSC_noPCA_res(repeat,:)=Clustering8Measure(Y,label);
end
TLMSC_noPCA_res=mean(TLMSC_noPCA_res);
fprintf('TLMSC without PCA result: ACC %f   NMI %f \n',TLMSC_noPCA_res(1)*100,TLMSC_noPCA_res(2)*100)
%%%%%%%%%%%%%%%%%%%%% run TLMSC with UMAP %%%%%%%%%%%%%%%%%%%%%%%%
load(['./para/',datasetName,'_TLMSC_UMAP']);%load parameters
X=XX;
for v=1:length(X)%use UMAP
    dim=size(X{v},2);
    if dim > DIM_PCA
        X{v} = py.umap.UMAP(pyargs('min_dist',int64(0),'n_components',int64(DIM_PCA),'n_neighbors',int64(umapnn))).fit_transform(X{v});
        X{v}=double(X{v});
    end
    X{v}=normalize(X{v},1);
end
for repeat=1:REPEAT
    em=transformone(X,neidian,cls_num,n_random);%Obtain improved low-dimensional embeddings
    label = fastSpectralClustering(em,cls_num,numflag);
    TLMSC_UMAP_res(repeat,:)=Clustering8Measure(Y,label);
end
TLMSC_UMAP_res=mean(TLMSC_UMAP_res);
fprintf('TLMSC with UMAP result: ACC %f   NMI %f \n',TLMSC_UMAP_res(1)*100,TLMSC_UMAP_res(2)*100)

%【By replacing the following sentence and tuning the parameter 'n_neighbors', Isomap and LLE also can be tested.】

% X{v} =py.sklearn.manifold.Isomap(pyargs('n_components',int64(DIM_PCA),'n_neighbors',int64(8))).fit_transform(X{v});
% X{v} =py.numpy.ascontiguousarray(X{v});
% X{v}=double(X{v});


% X{v} =py.sklearn.manifold.LocallyLinearEmbedding(pyargs('n_components',int64(DIM_PCA),'n_neighbors',int64(40))).fit_transform(X{v});
% X{v} =py.numpy.ascontiguousarray(X{v});
% X{v}=double(X{v});

