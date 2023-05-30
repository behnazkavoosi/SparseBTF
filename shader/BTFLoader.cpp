#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "structs.h"
#include <sutil/Exception.h>
#include "BTFLoader.h"
#include <math.h>
#include <stdio.h>

double ConvertNumberToFloat(unsigned int number)
{
    int mantissaShift = 10;
    unsigned long exponentMask = 0x7c00;
    int bias = 15;
    int signShift = 15;

    int sign = (number >> signShift) & 0x01;
    int exponent = ((number & exponentMask) >> mantissaShift) - bias;

    int power = -1;
    double total = 0.0;
    for (int i = 0; i < mantissaShift; i++)
    {
        int calc = (number >> (mantissaShift - i - 1)) & 0x01;
        total += calc * pow(2.0, power);
        power--;
    }
    double value = (sign ? -1 : 1) * pow(2.0, exponent) * (total + 1.0);

    return value;
}
using namespace std;

 //-----------------------------------------------------------------------------
 //  
 //  BTFLoader class definition
 //
 //-----------------------------------------------------------------------------

extern LaunchParams launch_params;

namespace
{
  cudaTextureObject_t create_texture(void* hostptr, unsigned int width, unsigned int height)
  {
    // Allocate CUDA array in device memory
    int32_t               pitch = width * 4 * sizeof(unsigned char);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    //channel_desc.f = cudaChannelFormatKindSigned;

    cudaArray_t cuda_array = nullptr;
    CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, hostptr, pitch, pitch, height, cudaMemcpyHostToDevice));

    // Create texture object
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    cudaTextureDesc tex_desc = {};

    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;   // 1
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 0;
    tex_desc.sRGB = 0;

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    return cuda_tex;
  }

  cudaTextureObject_t create_texture_float(void* hostptr, unsigned int width, unsigned int height)
  {

    // Allocate CUDA array in device memory
    int32_t               pitch = width * 4 * sizeof(unsigned char);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);//cudaCreateChannelDesc<uchar4>();
    //channel_desc.f = cudaChannelFormatKindSigned;

    cudaArray_t cuda_array = nullptr;
    CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, hostptr, pitch, pitch, height, cudaMemcpyHostToDevice));

    // Create texture object
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    cudaTextureDesc tex_desc = {};

    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.addressMode[2] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;  //cudaReadModeNormalizedFloat
    tex_desc.normalizedCoords = 1;   // 1
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 89;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 0;
    tex_desc.sRGB = 0;

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    return cuda_tex;
  }

  cudaTextureObject_t create_texture_float_3d(void* hostptr, unsigned int width, unsigned int height, unsigned int depth)
  {
    cudaArray_t d_volumeArray = nullptr;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    CUDA_CHECK(cudaMalloc3DArray(&d_volumeArray, &channelDesc, make_cudaExtent(width, height, depth), cudaArrayDefault));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void*)hostptr, width * sizeof(float), width, height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent = make_cudaExtent(width, height, depth);
    copyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_volumeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    // access with normalized texture coordinates
    texDescr.normalizedCoords = 0;
    // linear interpolation
    texDescr.filterMode = cudaFilterModePoint;
    // wrap texture coordinates
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &texRes, &texDescr, NULL));
    return cuda_tex;
  }
}

void BTFLoader::load_btf_coeff_files(const std::string& filename, unsigned int sparsity, unsigned int channel)
{
  LaunchParams& lp = launch_params;

  ifstream file(filename.c_str(), std::ifstream::binary);
  if(!file)
  {
    cout << "File not found: " << filename << endl;
    exit(0);
  }

  // get its size
  size_t fileSize;
  file.seekg(0, std::ios::end);
  fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // read the data
  vector<uint16_t> fileData(fileSize);
    file.read((char*)&fileData[0], fileSize);
    
    for (unsigned int i = 0; i < 4; ++i)
        *(&lp.btf[channel].dims.x + i) = static_cast<unsigned int>(fileData[i]);
    
    unsigned int num_patch = 400 * 400 / (lp.btf[channel].dims.x * lp.btf[channel].dims.y); // number of patches
    lp.btf[channel].NNZs = sparsity * num_patch;                    // total number of non-zero values 
    lp.btf[channel].sparsity = sparsity;

    vector<float> h_LNZ1(lp.btf[channel].NNZs);
    vector<float> h_LNZ2(lp.btf[channel].NNZs);
    vector<float> h_LNZ3(lp.btf[channel].NNZs);
    vector<float> h_LNZ4(lp.btf[channel].NNZs);
    vector<float> h_Memb(num_patch);
    vector<float> h_NNZ(num_patch);
    vector<float> h_NZ2DCoeffvar(lp.btf[channel].NNZs);
    
    for (unsigned int j = 0; j < lp.btf[channel].NNZs; ++j)
    {
        h_LNZ1[j] = static_cast<float>(fileData[0 + 4 * (j + 1)]);
        h_LNZ2[j] = static_cast<float>(fileData[1 + 4 * (j + 1)]);
        h_LNZ3[j] = static_cast<float>(fileData[2 + 4 * (j + 1)]);
        h_LNZ4[j] = static_cast<float>(fileData[3 + 4 * (j + 1)]);
        h_NZ2DCoeffvar[j] = static_cast<float>(ConvertNumberToFloat(fileData[j + lp.btf[channel].NNZs * 4 + 4 + num_patch * 2]));
    }

    for (unsigned int j = 0; j < num_patch; ++j)
    {
        h_Memb[j] = static_cast<float>(fileData[j + lp.btf[channel].NNZs * 4 + 4]);
        h_NNZ[j] = static_cast<float>(fileData[j + lp.btf[channel].NNZs * 4 + 4 + num_patch]);
    }

  m_samplers.push_back(create_texture_float(reinterpret_cast<void*>(&h_NNZ[0]), num_patch, 1));
  lp.btf[channel].tx_NNZ = m_samplers.back();
  m_samplers.push_back(create_texture_float(reinterpret_cast<void*>(&h_Memb[0]), num_patch, 1));
  launch_params.btf[channel].tx_Memb = m_samplers.back();
  m_samplers.push_back(create_texture_float(reinterpret_cast<void*>(&h_LNZ1[0]), sparsity, num_patch));
  launch_params.btf[channel].tx_LNZ1 = m_samplers.back();
  m_samplers.push_back(create_texture_float(reinterpret_cast<void*>(&h_LNZ2[0]), sparsity, num_patch));
  launch_params.btf[channel].tx_LNZ2 = m_samplers.back();
  m_samplers.push_back(create_texture_float(reinterpret_cast<void*>(&h_LNZ3[0]), sparsity, num_patch));
  launch_params.btf[channel].tx_LNZ3 = m_samplers.back();
  m_samplers.push_back(create_texture_float(reinterpret_cast<void*>(&h_LNZ4[0]), sparsity, num_patch));
  launch_params.btf[channel].tx_LNZ4 = m_samplers.back();
  m_samplers.push_back(create_texture_float(reinterpret_cast<void*>(&h_NZ2DCoeffvar[0]), sparsity, num_patch));
  launch_params.btf[channel].tx_NZ2DCoeffvar = m_samplers.back();
}

void BTFLoader::load_btf_dict_files(const std::string& filename, unsigned int num_dict, unsigned int channel)
{ /*
  if(dims.size() < 4)
  {
    cout << "Load coefficients before dictionary to get the dictionary dimensions." << endl;
    exit(0);
  } */

  LaunchParams& lp = launch_params;

  ifstream file(filename.c_str(), std::ifstream::binary);
  if(!file)
  {
    cout << "File not found: " << filename << endl;
    exit(0);
  }

  // get its size
  size_t fileSize;
  file.seekg(0, ios::end);
  fileSize = static_cast<size_t>(file.tellg());
  file.seekg(0, ios::beg);

  // read the data
  vector<uint16_t> fileData(fileSize);
    file.read((char*)&fileData[0], fileSize);

    size_t dict1_s = lp.btf[channel].dims.x * lp.btf[channel].dims.x * num_dict;
    size_t dict2_s = lp.btf[channel].dims.y * lp.btf[channel].dims.y * num_dict;
    size_t dict3_s = lp.btf[channel].dims.z * lp.btf[channel].dims.z * num_dict;
    size_t dict4_s = lp.btf[channel].dims.w * lp.btf[channel].dims.w * num_dict;

    vector<float> h_EnsembleDic1(dict1_s);
    vector<float> h_EnsembleDic2(dict2_s);
    vector<float> h_EnsembleDic3(dict3_s);
    vector<float> h_EnsembleDic4(dict4_s);
    
    for (unsigned int j = 0; j < dict1_s; ++j)
        h_EnsembleDic1[j] = static_cast<float>(ConvertNumberToFloat(fileData[j]));

    for (unsigned int j = 0; j < dict2_s; ++j)
        h_EnsembleDic2[j] = static_cast<float>(ConvertNumberToFloat(fileData[j + dict1_s]));

    for (unsigned int j = 0; j < dict3_s; ++j)
        h_EnsembleDic3[j] = static_cast<float>(ConvertNumberToFloat(fileData[j + dict1_s + dict2_s]));

    for (unsigned int j = 0; j < dict4_s; ++j)
        h_EnsembleDic4[j] = static_cast<float>(ConvertNumberToFloat(fileData[j + dict1_s + dict2_s + dict3_s]));

  m_samplers.push_back(create_texture_float_3d(reinterpret_cast<void*>(&h_EnsembleDic1[0]), lp.btf[channel].dims.x, lp.btf[channel].dims.x, num_dict));
  lp.btf[channel].tx_EnsembleDic1 = m_samplers.back();
  m_samplers.push_back(create_texture_float_3d(reinterpret_cast<void*>(&h_EnsembleDic2[0]), lp.btf[channel].dims.y, lp.btf[channel].dims.y, num_dict));
  lp.btf[channel].tx_EnsembleDic2 = m_samplers.back();
  m_samplers.push_back(create_texture_float_3d(reinterpret_cast<void*>(&h_EnsembleDic3[0]), lp.btf[channel].dims.z, lp.btf[channel].dims.z, num_dict));
  lp.btf[channel].tx_EnsembleDic3 = m_samplers.back();
  m_samplers.push_back(create_texture_float_3d(reinterpret_cast<void*>(&h_EnsembleDic4[0]), lp.btf[channel].dims.w, lp.btf[channel].dims.w, num_dict));
  lp.btf[channel].tx_EnsembleDic4 = m_samplers.back();
}

void BTFLoader::load_btf_adj_map(const std::string& filename)
{
    LaunchParams& lp = launch_params;

    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // read the data:
    std::vector<int16_t> fileData(fileSize);
    file.read((char*)&fileData[0], fileSize);

    size_t width = 256;
    size_t height = 1024;
    size_t depth = 3;
    size_t size = width * height * depth;
    vector<float> h_Map(size);

    for (unsigned int j = 0; j < size; ++j)
        h_Map[j] = static_cast<float>(fileData[j]);

    m_samplers.push_back(create_texture_float_3d(reinterpret_cast<void*>(&h_Map[0]), width, height, depth));
    lp.adj_map = m_samplers.back();

    //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.adj_map), h_Map.size() * sizeof(float)));
    //CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.adj_map), reinterpret_cast<void*>(&h_Map[0]), h_Map.size() * sizeof(float), cudaMemcpyHostToDevice));

}

void BTFLoader::load_btf_mean_std(const std::string& filename) const
{
  LaunchParams& lp = launch_params;

  std::streampos fileSize;
  std::ifstream file(filename, std::ios::binary);

  // get its size:
  file.seekg(0, std::ios::end);
  fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // read the data:
  std::vector<double> fileData(fileSize);
  file.read((char*)&fileData[0], fileSize);

  size_t size = 22801;
  vector<float> h_Mean(size);
  for(unsigned int j = 0; j < h_Mean.size(); ++j)
    h_Mean[j] = static_cast<float>(fileData[j]);

  vector<float> h_Std(size);
  for(unsigned int j = 0; j < h_Std.size(); ++j)
    h_Std[j] = static_cast<float>(fileData[j + h_Mean.size()]);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.mean), h_Mean.size()*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.mean), reinterpret_cast<void*>(&h_Mean[0]), h_Mean.size()*sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.std), h_Std.size()*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.std), reinterpret_cast<void*>(&h_Std[0]), h_Std.size()*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void BTFLoader::load_btf_angles_list(const std::string& filename) const
{
  LaunchParams& lp = launch_params;

  std::streampos fileSize;
  std::ifstream file(filename, std::ios::binary);

  // get its size:
  file.seekg(0, std::ios::end);
  fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // read the data:
  std::vector<double> fileData(fileSize);
  file.read((char*)&fileData[0], fileSize);

  size_t size = 151;
  vector<float> h_Phi(size);
  vector<float> h_Theta(size);
  vector<unsigned int> h_Patch(size);

  for(size_t j = 0; j < size; ++j)
  {
    h_Phi[j] = static_cast<float>(fileData[j]);
    h_Theta[j] = static_cast<float>(fileData[h_Phi.size() + j]);
    h_Patch[j] = static_cast<unsigned int>(fileData[h_Phi.size() + h_Theta.size() + j]);
  }

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.phi), h_Phi.size()*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.phi), reinterpret_cast<void*>(&h_Phi[0]), h_Phi.size()*sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.theta), h_Theta.size()*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.theta), reinterpret_cast<void*>(&h_Theta[0]), h_Theta.size()*sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.ang_patch), h_Patch.size()*sizeof(unsigned int)));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.ang_patch), reinterpret_cast<void*>(&h_Patch[0]), h_Patch.size()*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void BTFLoader::load_btf_loc_list(const std::string& filename) const
{
    LaunchParams& lp = launch_params;

    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // read the data:
    std::vector<double> fileData(fileSize);
    file.read((char*)&fileData[0], fileSize);

    size_t size = 151;
    vector<float> h_x(size);
    vector<float> h_y(size);
    vector<float> h_z(size);
    vector<unsigned int> h_Patch(size);

    for (size_t j = 0; j < size; ++j)
    {
        h_x[j] = static_cast<float>(fileData[j]);
        h_y[j] = static_cast<float>(fileData[h_x.size() + j]);
        h_z[j] = static_cast<float>(fileData[h_x.size() + h_y.size() + j]);
        h_Patch[j] = static_cast<unsigned int>(fileData[h_x.size() + h_y.size() + h_z.size() + j]);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.x), h_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.x), reinterpret_cast<void*>(&h_x[0]), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.y), h_y.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.y), reinterpret_cast<void*>(&h_y[0]), h_y.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.z), h_z.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.z), reinterpret_cast<void*>(&h_z[0]), h_z.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.ang_patch), h_Patch.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.ang_patch), reinterpret_cast<void*>(&h_Patch[0]), h_Patch.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

}

void BTFLoader::cleanup()
{
  // Destroy textures (base_color, metallic_roughness, normal)
  for(cudaTextureObject_t& texture : m_samplers)
    CUDA_CHECK(cudaDestroyTextureObject(texture));
  m_samplers.clear();

  if(loaded_means_angles)
  {
    LaunchParams& lp = launch_params;
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lp.mean)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lp.std)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lp.phi)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lp.theta)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lp.x)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lp.y)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lp.z)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lp.ang_patch)));
  }
}
