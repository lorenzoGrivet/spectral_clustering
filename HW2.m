clc;
clear;
format long;

% Here we open the files and we store the data in separate matrices
spiral_ds=load("Spiral.mat").X;
circle_ds=load("Circle.mat").X;

%spiegare perché scelta soglia
thresh_c=0.01;
thresh_s=0.001;

cl_spiral=main(spiral_ds, 10, 20,thresh_s,"Spiral");
cl_circle=main(circle_ds, 10, 20,thresh_c,"Circle");

if check_clusters(spiral_ds,cl_spiral)
    disp("Cluster (spiral) corretti");
else
    disp("Cluster (spiral) non corretti");
end



% main function that runs all
function clusters=main(ds, k, n_eigen, threshold, name)
    S = similarity_matrix(ds,1);  % construction of the similarity matrix
    W = knn(S, k); % using knn algorithm we compute the adjacency matrix
    D = degreeMatrix(W); % construction of the degree matrix
    L = D - W; % Laplacian matrix 
    % W, D, L matrices are stored in sparse format
    
    % computation of the eigenvalues and eigenvectors
    [eigenvectors, eigenvaluesMatrix] = eigs(L, n_eigen, 'smallestabs');
    figure;
    eigenvalues = diag(eigenvaluesMatrix);
    % plot of the n smallest eigenvalues,
    semilogy(eigenvalues, '-o', 'MarkerSize', 5, 'Color', 'b');
    xlabel('Eigenvalues');
    ylabel('Value')
    title(sprintf('First %d eigenvalues. %s', n_eigen,name))
    
    % we only consider the eigevalues closest to 0 by setting a treshold of 0.01.
    % The matrix U is then computed using their corresponding eigenvectors.
    
    n_clusters = nnz(eigenvalues <= threshold);


    U = eigenvectors(:, 1:n_clusters);
    
    clusters = kmeans(U, n_clusters);
    

    % scatter plot of the dataset points with respect to the clusters
    % computed by the kmeans algrithm 
    figure;
    scatter(ds(:,1), ds(:,2), 15, clusters, 'filled')
    colormap(jet);
    xlabel('X')
    ylabel('Y')
    title(sprintf('%s dataset clusters', name))

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
                if v > 1e-7  % threshold set to take into account only the points with significative distance
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

function b=check_clusters(ds,cl)
    %check if clusters we found are the same of the ones in the third
    %column of the file spiral.mat indipendentemente of the order
    %********************************************************************* 
    % ******************************************************************************
    [m,n] =size(ds);
    e=1;
    f=1;
    if n>2
        l=ds(:,3);
        c1=[];
        c2=[];
        
        for i=2:m
            if l(i)==l(i-1)
                c1=[c1,e];
            else
                e=e+1;
                c1=[c1,e];
            end


            if cl(i)==cl(i-1)
                c2=[c2,f];
            else
                e=e+1;
                c2=[c2,f];
            end
        end

        b=norm(c1-c2,2);
        

    end
end
