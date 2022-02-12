clc;
clear all;

I_t=imread('D:\Sumana\code\data\QF3_90\2.jpg'); %triple compressed
%I_t = I_t+48 ;
I_d=imread('D:\Sumana\code\data\QF2_50\2.jpg'); %double compressed
%I_d = I_d+48 ;

%I2=imcrop(I,[75 68 130 112]);
%[J, rect] = imcrop(I);
%I3=imcrop(I,[96 53 340 278]);
%D=imread('6_double.jpg');

%50 percent forgery

%actual Ic=imcrop(I2,[85 63 370 250]);
Ic=imcrop(I_t,[82 45 361 293]);
A=I_d;
B=Ic;
B=padarray(B,[1 1]); %If you want borders
Y=85;
X=63;
A((1:size(B,1))+X,(1:size(B,2))+Y,:) = B;
imshow(A);
imwrite(A,'forged.jpg','jpg','Quality',100);

%20 percent forgery
Ic=imcrop(I_t,[75 76 214 237]);
A=I_d;
B=Ic;
B=padarray(B,[1 1]); %If you want borders
Y=75;
X=75;
A((1:size(B,1))+X,(1:size(B,2))+Y,:) = B;
imshow(A);
imwrite(A,'D:\Sumana\code\triple\90-50-90.jpg','jpg','Quality',100);

%10 percent forgery;
Ic=imcrop(I_t,[171 111 149 148]);
A=I_d;
B=Ic;
B=padarray(B,[1 1]); %If you want borders
Y=171;
X=111;
A((1:size(B,1))+X,(1:size(B,2))+Y,:) = B;
imshow(A);
imwrite(A,'forged.jpg','jpg','Quality',100);