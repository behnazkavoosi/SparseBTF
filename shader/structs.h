#pragma once

#include <optix.h>
#include "sutil/vec_math.h"
#include "sutil/Matrix.h"
#include <cuda/BufferView.h>
#include <cuda/GeometryData.h>
#include <texture_types.h>

//#define PASS_PAYLOAD_POINTER

using namespace sutil;

struct PositionSample
{
  float3 pos;
  float3 dir;
  float3 normal;
  float3 L;
};

struct BTF
{
  uint4               dims;
  unsigned int        NNZs;
  unsigned int        sparsity;

  cudaTextureObject_t tx_EnsembleDic1;
  cudaTextureObject_t tx_EnsembleDic2;
  cudaTextureObject_t tx_EnsembleDic3;
  cudaTextureObject_t tx_EnsembleDic4;

  cudaTextureObject_t tx_NNZ;
  cudaTextureObject_t tx_LNZ1;
  cudaTextureObject_t tx_LNZ2;
  cudaTextureObject_t tx_LNZ3;
  cudaTextureObject_t tx_LNZ4;
  cudaTextureObject_t tx_Memb;
  cudaTextureObject_t tx_NZ2DCoeffvar;
};

struct LaunchParams
{
  float*                   mean;
  float*                   std;
  float*                   phi;
  float*                   theta;
  float*				           x;
  float*                   y;
  float*				           z;
  cudaTextureObject_t	     adj_map;  //unsigned int*
  unsigned int*            ang_patch;
  BTF                      btf[3];
  float                    tex_scale;
};


