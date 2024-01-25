function [newtri] = tridelete(tri,nei,cls_num,n_inliers,inputs,n_random)
N=length(tri{1});
V=length(tri);
NN=length(nei{1});

hardp=floor(NN/cls_num);
thisd=hardfu_triplets_d(hardp,cls_num,floor(NN/cls_num));
for v=1:V
    if n_random~=0
    if NN>9999
    inputs{v}=py.sklearn.decomposition.TruncatedSVD(pyargs('n_components',int64(10),'random_state',int64(0))).fit_transform(inputs{v});
    inputs{v}=double(inputs{v});
    inputs{v}=normalize(inputs{v},1);
    end
    hardD = pdist2(inputs{v}, inputs{v}, 'euclidean');
    end
    t{v}=1;
    compare=[];
    newtri{v}=zeros(N,3);
    for i=1:N
        xx=linspace(1,V,V);
        xx=xx(xx~=v);
        maodian=tri{v}(i,1); zhengli=tri{v}(i,2);
        if mod(i,n_inliers)==1
            compare=cell(1,V-1);
            for vv=1:V-1
                compare{vv}=nei{xx(vv)}(maodian,:);
            end
        end
        flag=0;
        for vv=1:V-1
        flag=flag+sum(compare{vv}==zhengli);
        end
        if flag>0
            newtri{v}(t{v},:)=tri{v}(i,:);
            t{v}=t{v}+1;
            if flag>1&&n_random~=0
                new=reshape(randperm(NN,hardp*n_random),[n_random,hardp]);
                D=hardD(maodian,:);
                DD = D(new);
                [~,ind] = mink(DD,thisd,2);
                for r=1:n_random
                    newh=new(r,:);
                    newtri{v}(t{v},:)=[maodian,zhengli,newh(ind(r,thisd))];
                    t{v}=t{v}+1;
                end
            end
        end
    end
    newtri{v}=newtri{v}(1:t{v}-1,:);
end
end


