clc;
clear;
format long;


spiral_ds=load("Spiral.mat").X;
circle_ds=load("Circle.mat").X;
 


create_clusters(circle_ds, 10, 20)
create_clusters(spiral_ds, 10,20)


function create_clusters(ds, k, n_eigen)
    S = similarity_matrix(ds,1);
    W = knn(S, k);
    D = degreeMatrix(W);
    L = D - W;
    [eigenvectors, eigenvaluesMatrix] = eigs(L, n_eigen, 'smallestabs');
    
    
    figure;
    eigenvalues = diag(eigenvaluesMatrix);
    plot(eigenvalues, '-o', 'MarkerSize', 5, 'Color', 'b');
    xlabel('Eigenvalues');
    ylabel('Value')
    title(sprintf('First %d eigenvalues', n_eigen))
    
    
    
    threshold = 0.01;
    n_clusters = nnz(eigenvalues <= threshold);

    %we take the first n eigenvalues as they are the closest one to 0, given
    %that we compute the matrix U with their corresponding eigenvectors

    U = eigenvectors(:, 1:n_clusters);
    
    clusters = kmeans(U, n_clusters);
    
    figure;
    scatter(ds(:,1), ds(:,2), 15, clusters, 'filled')
    colormap(jet);
    xlabel('X')
    ylabel('Y')
    title('Dataset clusters')

end




function m = similarity_matrix(ds,sigma)
    [r,~]=size(ds);
    m=zeros(r,r);   
    for i=1:r
        for j=i:r
            
            if i==j
                m(i,j)=0;
            else
                v=f_sim(ds(i,1:2),ds(j,1:2),sigma);
                if v > 1e-7
                    m(i,j)=v;
                    m(j,i)=v;
                end 
            end
        end
    end

end



function s = f_sim(x1,x2,sigma)
    s=exp(-norm(x1 - x2)^2 / (2 * sigma^2));
end


function W = knn(S, k)
    [m,n] = size(S);
    M = zeros(m,n);
    
    for i = 1 : m
        [~, sortedIndices] = sort(S(i, :), 'descend');
        sortedIndices = sortedIndices(1 : k);
        for j = 1 : length(sortedIndices)
            M(i,sortedIndices(j)) = S(i,sortedIndices(j));
            M(sortedIndices(j), i) = S(i,sortedIndices(j));
        end
    end

    if M == M'
        W = sparse(M);
    end
end


function D = degreeMatrix(W)
    degrees = sum(W, 2); 
    D = spdiags(degrees, 0, size(W, 1), size(W, 1));
end

