%imgs = dir('*.png');
dataset_dir = 'SR_testing_datasets/Set5';
%dataset_dir = 'SR_testing_datasets/Set14';
%dataset_dir = 'SR_testing_datasets/Urban100';

imgs = dir(strcat(dataset_dir,'/','*.png'));

lr_output_dir = strcat(dataset_dir,'/','LR');
hr_output_dir = strcat(dataset_dir,'/','HR');
if exist(lr_output_dir,'dir')==7
    %rmdir(lr_output_dir);
    command = strcat("rm -rf ",lr_output_dir);
    system(command);
end
if exist(hr_output_dir,'dir')==7
    %rmdir(hr_output_dir);
    command = strcat("rm -rf ",hr_output_dir);
    system(command);
end
mkdir(lr_output_dir);
mkdir(hr_output_dir);
%mkdir LR;
%mkdir HR;

%for i = 0:119
for i = 0:(length(imgs)-1)
    disp(i);
    %cropped = imread(imgs(i+1).name);
    cropped = imread(strcat(dataset_dir,'/',imgs(i+1).name));
    
    %crop 4 border pixels on top and bottom to be divisable by 16
    %do not crop for AGD, TrilevelNAS
    %cropped = cropped(5:end-4,:,:);
    
    disp(size(cropped));
    
    %x16 = imresize(cropped, 0.0625);
    img_lr = imresize(cropped, 0.25);
    
    size(img_lr);
    
    %imwrite(img_lr,strcat('LR/', pad(num2str(i, 3), 6, 'left', '0') , '.png'));
    %imwrite(cropped,strcat('HR/', pad(num2str(i, 3), 6, 'left', '0') , '.png'));
    imwrite(img_lr,strcat(lr_output_dir,'/', pad(num2str(i, 3), 6, 'left', '0') , '.png'),'png');
    imwrite(cropped,strcat(hr_output_dir,'/', pad(num2str(i, 3), 6, 'left', '0') , '.png'),'png');
    
end