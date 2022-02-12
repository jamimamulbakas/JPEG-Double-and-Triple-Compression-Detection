clc;
clear all;
res = struct2array(load('res.mat'));
m=size(res,1);
n=size(res,2);
figure, imshow(res);
im1(1:m,1:n)=1;
for_blk=237;
for_blk2=223;
im1(77:77+for_blk-1,70:70+for_blk2-1)=0;
imshow(im1);
im2=res;
TP=0;
TN=0;
FP=0;
FN=0;
for i=1:m
    for j=1:n
        if im1(i,j)==1 && im2(i,j)==1 
            TP=TP+1;
        elseif im1(i,j)==0 && im2(i,j)==0 
            TN=TN+1;
        elseif (im1(i,j)==0 && im2(i,j)==1)
            FP=FP+1;
        elseif (im1(i,j)==1 && im2(i,j)==0)
            FN=FN+1;
        end
    end
end
  accuracy1 = (TP+TN)/(TP+TN+FP+FN);

  recall = TP/(TP+FP);
  precision = TP/(TP+FN);
  
  f1_score = (2 * (precision * recall)) / (precision + recall);
  disp('accuracy');disp(accuracy1);
  disp('f1_score'); disp(f1_score);