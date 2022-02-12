
close all
clear
clc
clc;
clear all;

stride = 8;
block = 62;
y=zeros(384,512);
probMatrix = struct2array(load('90_50_90\multi_pM_50_70_90.mat'));
hor_ctr = 0;
ver_ctr = 0;
hor_length = size(y,1)-61;
ver_length = size(y,2)-61;
for j = 1:stride:324
    hor_ctr = hor_ctr+1;
    for k = 1:stride:449
        ver_ctr = ver_ctr+1;
        if(ver_ctr<=57&&hor_ctr<=41)
        y(j:j+block-1,k:k+block-1) = probMatrix(hor_ctr,ver_ctr);
        end
    end
    ver_ctr = 0;
end
figure(1)
imagesc(y);
colorbar
fig=figure(1);
colorbar
mycmap = get(fig,'Colormap');
set(fig,'Colormap',flipud(mycmap))
imagesc(y);

P = impixel(y,map);

image = mat2gray(y);

cMap = hsv(256); % Whatever colormap you want.
rgbImage = ind2rgb(y, cMap);
imwrite(filename, rgbImage)
imshow(rgbImage)

%imwrite(image,'F:\pre-proc\fd\Results\CB\1.bmp');
GT_DatasetRoot = '.\gd\GroundTruth'; % Ground Truth
GT_DatasetRoot_extension = strcat(GT_DatasetRoot, '\*.bmp'); % image file extension
GT_Dataset = dir(GT_DatasetRoot_extension);

PR_DatasetRoot = '.\fd\Results\CB'; % The root of Predicted Results before applying post-processing
PR_DatasetRoot_extension = strcat(PR_DatasetRoot, '\*.bmp'); % image file extension
PR_Dataset = dir(PR_DatasetRoot_extension);

OutRoot = '.\PP_Results'; % destination root
OutExtension = 'bmp'; % Output Extension

Ni = length(PR_Dataset); % Number of images in dataset
s=1;
    GTImageName = strcat(GT_DatasetRoot, '\', GT_Dataset(s).name);
    GT = imread(GTImageName); % Ground Truth
    GT = im2bw(GT);
   
    
    PRImageName = strcat(PR_DatasetRoot, '\', PR_Dataset(s).name);
    PR = imread(PRImageName); % Predicted Result
    PR = im2bw(PR);
  
    
    [Hs, Ws] = size(PR);
    Area_PR = Hs*Ws;
    
    Bd = strel('square', 19);
    PRdilate = imdilate(PR, Bd);
  
    
    n = 8; % Connectivity
    L = bwlabel(PRdilate, n);
    Lv = L(:); % vectorized label
    Ncc = max(Lv); % the number of connected components
    Alpha = 0.01; % a constant
    PRdilate_filt = PRdilate; % initialization for the filtered PR
    for k = 1:1:Ncc
        Area_L = length(find(Lv == k));
        thr = Alpha*Area_PR; % a threshold
        if Area_L < thr
            for y = 1:1:Hs
                for x = 1:1:Ws
                    if L(y, x) == k
                        PRdilate_filt(y, x) = 2;
                    end
                end
            end
        end
    end
    figure, imshow(PRdilate_filt, [])

    PRdilate_filt_fill = imfill(PRdilate_filt, 'holes');
    %figure, imshow(PRdilate_filt_fill, [])
  save('res.mat','PRdilate_filt');
	
