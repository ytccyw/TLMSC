function [triplets] = sample_knn_triplets(neighbors, n_inliers)
n_points=size(neighbors,1);
anchors=repmat((1:n_points)',1,(n_inliers-1));
anchors=anchors';anchors=anchors(:);

inliers=repmat(neighbors(:,2:n_inliers),1,1);
inliers=inliers';inliers=inliers(:);

neighbors=neighbors(:,1:n_inliers);

outliers = getsample((n_inliers-1),n_points,neighbors);
outliers=outliers';outliers=outliers(:);
triplets=[anchors,inliers,outliers];
end

