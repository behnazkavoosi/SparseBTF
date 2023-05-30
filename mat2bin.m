% Convert .mat files to binary to use in the shader

function mat2bin(filename)

addpath('./Bonn-btflib/matlab/half_precision')  % https://github.com/cgbonn/btflib
load(sprintf('%s.mat', filename));
fid = fopen(sprintf('%s.bin', filename), 'w');

e = fwrite(fid, dims, 'int16');
e = fwrite(fid, LNZ, 'int16');
e = fwrite(fid, M, 'int16');
e = fwrite(fid, NNZ, 'int16');
e = fwrite(fid, halfprecision(NZ), 'uint16');

fclose(fid);

end
