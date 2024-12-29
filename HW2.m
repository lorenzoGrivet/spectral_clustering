clc;
clear;
format long;

spiral_ds=load("Spiral.mat").X;
circle_ds=load("Circle.mat").X;
 
S = mat(circle_ds,1);


W = knn(S, 10);
D = degreeMatrix(W);
L = D - W;

% G = graph(W);
% [bin, numComponents] = conncomp(G);


k = 20; 
opts.tol = 1e-10; % Tolleranza
opts.maxit = 1000; % Numero massimo di iterazioni


[eigenvectors, eigenvaluesMatrix] = eigs(L, k, 'smallestabs', opts);


figure;
eigenvalues = diag(eigenvaluesMatrix);
plot(eigenvalues, '-o', 'MarkerSize', 5, 'Color', 'b');




%usiamo i primi 3 autovalori in quanto sono i più vicini a 0, di
%conseguenza costruiamo la matrice U con i primi 3 autovettori

U = eigenvectors(:, 1:3);

clusters = kmeans(U, 3);

figure;
scatter(circle_ds(:,1),circle_ds(:,2),15,clusters, 'filled')
colormap(jet);




function m = mat(ds,sigma)
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

