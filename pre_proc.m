clear all;
close all;
clc;
s = 8;
b_size = 64;
range=11;
blkvalues=[1,2;2,1;3,1;2,2;1,3;1,4;2,3;3,2;4,1;5,1;4,2;3,3;2,4;1,5;1,6;2,5;3,4;4,3;5,2;6,1;7,1];
coeff=21;
%imag = imread('F:\pre-proc\test_image1\d1\1.jpg');
im = imread('D:\Sumana\code\data\60_70_80_90\QF4_90\1.jpg');
p1 = ((size(im,1)-b_size)/s)+1;
p2 = ((size(im,2)-b_size)/s)+1;
out = zeros(p1*p2*1,range*coeff);
blocknum=1;
for i =401:500
    clearvars -except i s b_size range blkvalues coeff im p1 p2 out blocknum
    close all force
    tic;
   % image = imread(strcat('F:\pre-proc\test_image1\d1\',int2str(i),'.jpg'));
    image =imread(strcat('D:\Sumana\code\data\60_70_80_90\QF4_90\',int2str(i),'.jpg'));
    YCBCR=rgb2ycbcr(image);
    Y = YCBCR(:,:,1);%image(:,:,1) + image(:,:,2) + image(:,:,3);
    x1 = size(Y,1)-b_size+1;
    y1 = size(Y,2)-b_size+1;
    for j = 1:s:x1
        for k = 1:s:y1
            blkimg = Y(j:j+b_size-1,k:k+b_size-1);
            
            arr=[];
            for m=1:8:b_size-7
                for n=1:8:b_size-7
                    blkimg2=blkimg(m:m+7,n:n+7);
                    freqblkimg=dct2(blkimg2);
                    arr1=[];
                    for l=1:coeff
                        arr1=[arr1, freqblkimg(blkvalues(l,1),blkvalues(l,2))];
                    end
                    arr=[arr;arr1];
                end
            end
            [maxvalues,I]=max(arr);
            imag=1;
            for m=1:coeff
                
                if I(m)<=(range+1)/2
                    out(blocknum,imag:imag+range-1)=arr(1:range,m);
                elseif I(m)>=b_size-(range+1)/2
                    out(blocknum,imag:imag+range-1)=arr(b_size-range+1:b_size,m);
                else
                    out(blocknum,imag:imag+range-1)=arr(I(m)-(range-1)/2:I(m)+(range-1)/2,m);
                end
                imag=imag+range;
            end
            blocknum=blocknum+1;
        end
    end
    toc;
    fprintf('Complete image %d\n',i);
end
save('D:\Sumana\code\data\60_70_80_90\QF4_90_401_500.mat','out')