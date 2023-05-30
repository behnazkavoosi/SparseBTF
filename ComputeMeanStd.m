%%%%%%%% This code computes mean and standard deviation of angular images
%%%%%%%% of a BTF and write each of them as a vector into a binary file.
%%%%%%%% This file is used in the rendering framework to de-normalize the
%%%%%%%% pixel value.

clear all
close all
clc

names = {'carpet11'}; 
savepath = './mean_std/';

for i = 1:length(names)

    clear im 
    clear cr_btf1
    clear cr_btf2
    clear cr_btf3
    clear imMean
    clear imStd

    names{i}
    load(sprintf('./%s_resampled_W400xH400_L151xV151.mat', names{i}));

    cr_btf1 = SxV * U1';
    cr_btf2 = SxV * U2';
    cr_btf3 = SxV * U3';

    im(:, :, 1) = 0.299 * cr_btf1 + 0.587 * cr_btf2 + 0.114 * cr_btf3;
    im(:, :, 2) = -0.14713 * cr_btf1 - 0.28886 * cr_btf2 + 0.436 * cr_btf3;
    im(:, :, 3) = 0.615 * cr_btf1 - 0.51499 * cr_btf2 - 0.10001 * cr_btf3;

    for j = 1:size(cr_btf1, 2)
    
        imMean(j, 1) = mean2(im(:, j, :));
        fStd1 = sqrt(sum((im(:, j, 1) - imMean(j, 1)).^2) / (size(cr_btf1, 1) - 1));
        fStd2 = sqrt(sum((im(:, j, 2) - imMean(j, 1)).^2) / (size(cr_btf1, 1) - 1));
        fStd3 = sqrt(sum((im(:, j, 3) - imMean(j, 1)).^2) / (size(cr_btf1, 1) - 1));

        imStd(j, 1) = (fStd1 + fStd2 + fStd3) ./ 3;
    end

    save(sprintf('%s%s_mean_std.mat', savepath, names{i}), 'imMean', 'imStd');

    fid = fopen(sprintf('%s%s_mean_std.bin', savepath, names{i}), 'w');
    e = fwrite(fid, imMean, 'double');
    e = fwrite(fid, imStd, 'double');
    fclose(fid);
end


