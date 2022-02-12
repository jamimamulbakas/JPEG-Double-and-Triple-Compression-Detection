% Compressed the images and dave it into jpeg
clc
close all
clear all

sin_path='E:\PHD\Sumana\code\data\60_70_80\QF1_60';
dob_path='E:\PHD\Sumana\code\data\60_70_80\QF2_70';
tri_path='E:\PHD\Sumana\code\data\60_70_80\QF3_80';
im_dir='E:\data\ucid.v2\image';
im_dir1=dir(fullfile(im_dir,'*.tif'));
im_name={im_dir1.name};
im_num=length(im_name);
QF1=60;
QF2=70;
QF3=80;
for i=1:500
   img=imread(fullfile(im_dir,im_name{i})); 
   name=sprintf('%d.jpg',i);
   %Uncomment only one at a time
   sin_name=fullfile(sin_path,name);
   imwrite(img,sin_name,'jpg','quality',QF1);
   img1=imread(sin_name);
   dob_name=fullfile(dob_path,name);
   imwrite(img1,dob_name,'jpg','quality',QF2);
   img2=imread(dob_name);
   tri_name=fullfile(tri_path,name);
   imwrite(img2,tri_name,'jpg','quality',QF3);
   
end