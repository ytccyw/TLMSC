function[embedding]=transformone(inputs,n_inliers,cls_num,n_random)
n_points=size(inputs{1},1);V=length(inputs);
parfor v=1:V
    if n_points>10000
        sig{v} = dosig(inputs{v});  %Fast calculation the scaling factors for neighbourhood similarity
    else
        sig{v} =0;
    end
end
for v=1:V
    [SS{v}]= nnxiangsixing(inputs{v},n_inliers,sig{v});%  Compute neighbourhood similarity,Calculate only for the range of nearest neighbours 'n_inliers'
end
for v=1:V
    Hnei{v} = neinei(SS{v},n_inliers);%  Get Nearest Neighbors
    tri{v} = sample_knn_triplets(Hnei{v}, n_inliers);% generate triplets
    Hnei{v}=Hnei{v}(:,2:end);
end
if V>2
    [tri] = tridelete(tri,Hnei,cls_num,n_inliers-1,inputs,n_random); % the Triplet Selection Strategy
end
for v=1:V
    [wei{v},tri{v}] = triplet_weights(tri{v},SS{v});%  Calculate weights
    wei{v}=wei{v}/max(wei{v});
end
for v=2:V
    inputs{1}=[inputs{1},inputs{v}];
    tri{1}=[tri{1};tri{v}];
    wei{1}=[wei{1};wei{v}];
end
triplets=[tri{1}];
weights=[wei{1}];
clear tri
clear wei
% After obtaining the triplets and their weights, the following improved low-dimensional embedding is obtained
% by momentum gradient descent under their guidance

INIT_SCALE = 10^(-5);%   scaling factor
INIT_MOMENTUM = 0.5;%   Initial momentum size
FINAL_MOMENTUM = 0.8;
SWITCH_ITER = 100;%   For the first _SWITCH_ITER times of iterations, use _INIT_MOMENTUM as the momentum; for the rest of the iterations, use _FINAL_MOMENTUM as the momentum.
n_iters=200;
n_dims=5;%  output dimension
lr=0.1;%  learning rate

embedding = pca(inputs{1}','NumComponents',n_dims);%Initialize the low-dimensional embedding
embedding = INIT_SCALE .*embedding;

clear inputs
vel=zeros(n_points,n_dims);
gain=ones(n_points,n_dims);
losss=0;
for itr=1:n_iters
    if itr > SWITCH_ITER%250
        gamma = FINAL_MOMENTUM;%0.8
    else
        gamma = INIT_MOMENTUM;%0.5
    end
    [grad]=grad_loss(embedding+gamma.*vel,weights,triplets);
    if itr >= 100 && length(triplets)>1e6 && mod(itr,10)==0
        lossnow=lossdisp(embedding+gamma.*vel,weights,triplets);
        cha=abs(lossnow-losss)/losss;
        losss=lossnow;
        fprintf('loss：%f itr: %f \n',cha,itr);
        if cha<0.05
            break
        end
    end
    [embedding,vel,gain]= update_embedding_dbd(embedding,grad,vel,gain,lr,gamma);
end
end
function[embedding,vel,gain]= update_embedding_dbd(embedding,grad,vel,gain,lr,gamma)
MIN_GAIN = 0.01;
INCREASE_GAIN = 0.3;
DAMP_GAIN = 0.9;
[n,dim]=size(embedding);
for i=1:n
    for d=1:dim
        if sign(vel(i,d))~=sign(grad(i,d))
            gain(i,d)=gain(i,d)+INCREASE_GAIN;
        else
            gain(i,d)=max(gain(i,d) * DAMP_GAIN, MIN_GAIN);
        end
        vel(i,d) = gamma * vel(i,d) - lr * gain(i,d) * grad(i,d);
        embedding(i,d) = embedding(i,d)+vel(i,d);
    end
end
end
function[grad]=grad_loss(embedding,weights,triplets)
[n,dim]=size(embedding);
n_triplets=size(triplets,1);
grad=zeros(n,dim);
for t=1:n_triplets
    i=triplets(t,1);j=triplets(t,2);k=triplets(t,3);
    y_ij=embedding(i,:)-embedding(j,:);
    y_ik=embedding(i,:)-embedding(k,:);
    d_ij=1+norm(y_ij).^2;d_ik=1+norm(y_ik).^2;
    w = weights(t) / (d_ij + d_ik)^2;
    gs = y_ij.*(d_ik * w);
    go = y_ik.*(d_ij * w);
    grad(i,:)=grad(i,:)+(gs - go);
    grad(j,:)=grad(j,:)-gs;
    grad(k,:)=grad(k,:)+go;
end
end
function[loss]=lossdisp(embedding,weights,triplets)
n_triplets=size(triplets,1);
loss=0;
for t=1:n_triplets
    i=triplets(t,1);j=triplets(t,2);k=triplets(t,3);
    y_ij=embedding(i,:)-embedding(j,:);
    y_ik=embedding(i,:)-embedding(k,:);
    d_ij=1+norm(y_ij).^2;d_ik=1+norm(y_ik).^2;
    loss = loss+weights(t)/ (1 + d_ik / d_ij);
end
loss=loss/n_triplets;
end