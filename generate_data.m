clear all
close all
clc
inp1 = load('d1_QF1_90.mat'); %single compressed
mat1 = struct2array(inp1);
size1 = size(mat1,1);
size2 = size1*3;
inp2 = load('d2_QF2_50.mat'); %double compressed
mat2 = struct2array(inp2);
inp3 = load('d3_QF3_90.mat'); %triple compressed
mat3 = struct2array(inp3);
mat = [mat1;mat2;mat3];
label = zeros(size2,3);
for i=1:size2
    disp(i)
    if i<=size1
        label(i,:) = [1 0 0];
    else
       if i<=(size1*2)
        label(i,:) = [0 1 0];
       else
           label(i,:) = [0 0 1];
       end
    end
end
merge_matrix = [mat label];
shuffled_matrix = merge_matrix(randperm(length(merge_matrix)),:);
train_data = shuffled_matrix(:,1:131);
train_label = shuffled_matrix(:,232:234);
save('t_data_90_50_90.mat','train_data','-v7.3'); %complete training data
save('l_data_90_50_90.mat','train_label','-v7.3'); %corresponding label data
