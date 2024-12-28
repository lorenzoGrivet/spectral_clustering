clc;
clear;

spiral_ds=load("Spiral.mat").X;
circle_ds=load("Circle.mat").X;


mat(circle_ds,1);



disp("nj")


function m = mat(ds,sigma)
    [r,~]=size(ds);
    m=zeros(r,r);   

    for i=1:r
        for j=i:r
            
            if i==j
                m(i,j)=0;
            else
                v=f_sim(ds(i,1:2),ds(j,1:2),sigma);
                m(i,j)=v;
                m(j,i)=v;
            end


        end
    end

end



function s = f_sim(x1,x2,sigma)
    s=exp(-norm(x1 - x2)^2 / (2 * sigma^2));
end
