#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include "structs.h"
#include "envmap.h"
#include "AreaLight.h"

using namespace sutil;

extern "C" {
    __constant__ LaunchParams launch_params;
}

#define DIRECT
#define INDIRECT

// Given spherical coordinates, where theta is the 
// polar angle and phi is the azimuthal angle, this
// function returns the corresponding direction vector
__inline__ __host__ __device__ float3 spherical_direction(float sin_theta, float cos_theta, float phi)
{
  float sin_phi = sinf(phi), cos_phi = cosf(phi);
  return make_float3(sin_theta*cos_phi, sin_theta*sin_phi, cos_theta);
}

// [Frisvad, Journal of Graphics Tools 16, 2012;
//  Max, Journal of Computer Graphics Techniques 6, 2017]
__inline__ __host__ __device__ void onb_consistent(const float3& n, float3& b1, float3& b2)
{
    if (n.z < -0.9999805689f) // Handle the singularity
    {
        b1 = make_float3(0.0f, -1.0f, 0.0f);
        b2 = make_float3(-1.0f, 0.0f, 0.0f);
        return;
    }
    const float a = 1.0f / (1.0f + n.z);
    const float b = -n.x * n.y * a;
    b1 = make_float3(1.0f - n.x * n.x * a, b, -n.x);
    b2 = make_float3(b, 1.0f - n.y * n.y * a, -n.y);
}


// with interpolation in coefficients space
extern "C" __global__ void __closesthit__btf()
{
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;

  const LaunchParams& lp = launch_params;
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
  if(depth > lp.max_depth)
  {
    prd->result = make_float3(0.0f);
    return;
  }
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
  if(depth > 0)
    return;
#endif

  // Retrieve hit info
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  const float3& x = geom.P;
  const float3 wo = -optixGetWorldRayDirection(); // -normalize(lp.W); // Use lp.W to have only one direction toward the observer
  const float2 texcoords = geom.UV*lp.tex_scale;
  float3 n = normalize(geom.N);
  float cos_theta_o = dot(n, wo);
  const float s = copysignf(1.0f, cos_theta_o);
  cos_theta_o *= s;
  n *= s;

  float3 E, wi;
  const float no_of_sources = static_cast<float>(lp.lights.count);
  const float bgc = length(lp.miss_color);
  const float prob = bgc/(no_of_sources + bgc);
  const float xi = rnd(t);
  if(xi < prob)
  {
    // Environment map importance sampling
    //*
    sample_environment(x, wi, E, t);
    const float cos_theta_i = dot(wi, n);
    if(cos_theta_i <= 0.0f)
      return;
    if(traceOcclusion(lp.handle, x, wi, tmin, tmax))
      return;
    E *= cos_theta_i/prob;
  }
  else
  {
    // Direct illumination
    const unsigned int idx = static_cast<unsigned int>(rnd(t)*no_of_sources);
    const Directional& light = lp.lights[idx];
    wi = -light.direction;
    const float cos_theta_i = dot(wi, n);
    if(dot(wi, n) <= 0.0f)
      return;
    // Shadow ray
    if(traceOcclusion(lp.handle, x, wi, tmin, tmax))
      return;
    E = light.emission*cos_theta_i*(no_of_sources + bgc);
  }

  // Texture wrapping (?)
  const float tex_s = texcoords.x; // geom.UV.x - floorf(geom.UV.x);
  const float tex_t = texcoords.y; // -geom.UV.y - floorf(-geom.UV.y);
  //const float tex_s = geom.UV.x*textureScale - 0.001f*(textureScale - 1.0f);
  //const float tex_t = (1.0f - geom.UV.y)* textureScale - 0.001f*(textureScale - 1.0f);

  // Change of basis to tangent space
  float3 ta, bi;
  onb_consistent(n, ta, bi);
  const float3 b1 = make_float3(ta.x, bi.x, n.x);
  const float3 b2 = make_float3(ta.y, bi.y, n.y);
  const float3 b3 = make_float3(ta.z, bi.z, n.z);
  const float3 t_wi = b1*wi.x + b2*wi.y + b3*wi.z;
  const float3 t_wo = b1*wo.x + b2*wo.y + b3*wo.z;

    // Get spherical coordinates of directions (adding epsilon to avoid special case)
    const float theta_i = fabsf(M_PI_2f - acosf(t_wi.z) + 1.0e-4f);
    const float phi_i = atan2f(t_wi.y, t_wi.x);
    const float2 li = make_float2(phi_i, theta_i);
    const float theta_o = fabsf(M_PI_2f - acosf(t_wo.z) + 1.0e-4f);
    const float phi_o = atan2f(t_wo.y, t_wo.x);
    const float2 cam = make_float2(phi_o, theta_o);

    // Init look-up variables
    bool b_coeffs_interp = true;


    int cam_indx[3]{ 0 }, light_indx[3]{ 0 };
    float3 cam_poses[3]{ 0.0f,0.0f,0.0f }, light_poses[3]{ 0.0f,0.0f,0.0f };

    float dist_cams[3]{ tmax,tmax ,tmax };
    float dist_lights[3]{ tmax,tmax ,tmax };


    ///========================== search through the list for closest points =================
    // find the 4 closest camera and light angles
    for (unsigned int i = 0; i < 151; ++i)
    {

        const float2 angle_list = make_float2(lp.phi[i], lp.theta[i]);
        const float3 bin_dir = spherical_direction(cosf(angle_list.y), sinf(angle_list.y), angle_list.x);
        const float cam_dist = length(t_wo - bin_dir);
        const float light_dist = length(t_wi - bin_dir);

        if (cam_dist < dist_cams[0])
        {
            dist_cams[2] = dist_cams[1];
            cam_indx[2] = cam_indx[1];
            dist_cams[1] = dist_cams[0];
            cam_indx[1] = cam_indx[0];
            dist_cams[0] = cam_dist;
            cam_indx[0] = i;
        }
        else if (cam_dist < dist_cams[1])
        {
            dist_cams[2] = dist_cams[1];
            cam_indx[2] = cam_indx[1];

            dist_cams[1] = cam_dist;
            cam_indx[1] = i;
        }
        else if (cam_dist < dist_cams[2])
        {
            dist_cams[2] = cam_dist;
            cam_indx[2] = i;
        }

        if (light_dist < dist_lights[0])
        {
            dist_lights[2] = dist_lights[1];
            light_indx[2] = light_indx[1];
            dist_lights[1] = dist_lights[0];
            light_indx[1] = light_indx[0];
            dist_lights[0] = light_dist;
            light_indx[0] = i;
        }
        else if (light_dist < dist_lights[1])
        {
            dist_lights[2] = dist_lights[1];
            light_indx[2] = light_indx[1];

            dist_lights[1] = light_dist;
            light_indx[1] = i;
        }
        else if (light_dist < dist_lights[2])
        {
            dist_lights[2] = light_dist;
            light_indx[2] = i;
        }
    }

    for (int i = 0; i < 2; i++)
    {
        cam_poses[i] = make_float3(lp.x[cam_indx[i]], lp.y[cam_indx[i]], lp.z[cam_indx[i]]);
        light_poses[i] = make_float3(lp.x[light_indx[i]], lp.y[light_indx[i]], lp.z[light_indx[i]]);

    }


    float light_lerp[2]{ 0.0f };
    float cam_lerp[2]{ 0.0f };

    cam_lerp[0] = 0.5f + 0.5f * (dist_cams[1] - dist_cams[0]) / dist_cams[1];
    cam_lerp[1] = 1.0f - cam_lerp[0];
    light_lerp[0] = 0.5f + 0.5f * (dist_lights[1] - dist_lights[0]) / dist_lights[1];
    light_lerp[1] = 1.0f - light_lerp[0];


    // =========================== Calculate the Barycentric interpolation ============================================
    float3 cam_v0 = cam_poses[1] - cam_poses[0], cam_v1 = cam_poses[2] - cam_poses[0], cam_v2 = t_wo - cam_poses[0];
    float cam_d00 = dot(cam_v0, cam_v0);
    float cam_d01 = dot(cam_v0, cam_v1);
    float cam_d11 = dot(cam_v1, cam_v1);
    float cam_d20 = dot(cam_v2, cam_v0);
    float cam_d21 = dot(cam_v2, cam_v1);
    float cam_denom = cam_d00 * cam_d11 - cam_d01 * cam_d01;
    float cam_v = (cam_d11 * cam_d20 - cam_d01 * cam_d21) / cam_denom;
    float cam_w = (cam_d00 * cam_d21 - cam_d01 * cam_d20) / cam_denom;
    float cam_u = 1.0f - cam_v - cam_w;

    float cam_coeffs[3]{ 0.0f };

    cam_coeffs[0] = cam_u;
    cam_coeffs[1] = cam_w;
    cam_coeffs[2] = cam_v;


    float3 light_v0 = light_poses[1] - light_poses[0], light_v1 = light_poses[2] - light_poses[0], light_v2 = t_wi - light_poses[0];
    float light_d00 = dot(light_v0, light_v0);
    float light_d01 = dot(light_v0, light_v1);
    float light_d11 = dot(light_v1, light_v1);
    float light_d20 = dot(light_v2, light_v0);
    float light_d21 = dot(light_v2, light_v1);
    float light_denom = light_d00 * light_d11 - light_d01 * light_d01;
    float light_v = (light_d11 * light_d20 - light_d01 * light_d21) / light_denom;
    float light_w = (light_d00 * light_d21 - light_d01 * light_d20) / light_denom;
    float light_u = 1.0f - light_v - light_w;


    float light_coeffs[3]{ 0.0f };

    light_coeffs[0] = light_u;
    light_coeffs[1] = light_w;
    light_coeffs[2] = light_v;

    const uint2 spatialPos = make_uint2(tex_s * 400, tex_t * 400); //each image in BTF is of size 400x400
    const uint2 patchLoc = make_uint2(spatialPos.x / 10, spatialPos.y / 10);
    const float2 spatialLoc = make_float2(spatialPos.x % 10, spatialPos.y % 10);
    unsigned int patchId = patchLoc.y * 40 + patchLoc.x;
    float patchIdx = static_cast<float>(patchId + 0.5f) / 1600.0f;  // normalize the index to use the wrap address mode (to be able to repeat BTF)

    float iNNZ_Y = tex1D<float>(lp.btf[0].tx_NNZ, patchIdx);
    float iMemb_Y = tex1D<float>(lp.btf[0].tx_Memb, patchIdx);

    float iNNZ_U = tex1D<float>(lp.btf[1].tx_NNZ, patchIdx);
    float iMemb_U = tex1D<float>(lp.btf[1].tx_Memb, patchIdx);

    float iNNZ_V = tex1D<float>(lp.btf[2].tx_NNZ, patchIdx);
    float iMemb_V = tex1D<float>(lp.btf[2].tx_Memb, patchIdx);

    float3 f_r = make_float3(0.0f, 0.0f, 0.0f);


    float std_vals[3][3]{ 0.0f };
    float mean_vals[3][3]{ 0.0f };

    for (int l_idx = 0; l_idx < 3; l_idx++) // light
        for (int c_idx = 0; c_idx < 3; c_idx++) // cam
        {
            std_vals[l_idx][c_idx] = lp.std[cam_indx[c_idx] * 151 + light_indx[l_idx]];
            mean_vals[l_idx][c_idx] = lp.mean[cam_indx[c_idx] * 151 + light_indx[l_idx]];
        }


    float meanvl = light_lerp[0] * (cam_lerp[0] * mean_vals[0][0] + cam_lerp[1] * mean_vals[0][1]) +
        light_lerp[1] * (cam_lerp[0] * mean_vals[1][0] + cam_lerp[1] * mean_vals[1][1]);

    //float meanvl = 0.5 * (0.5 * mean_vals[0][0] + 0.5 * mean_vals[0][1]) +
    //    0.5 * (0.5 * mean_vals[1][0] + 0.5 * mean_vals[1][1]);

    float3 colval{ 0.0f,0.0f,0.0f };

    //============================= Coeffs space interpolation =========================================

    if (b_coeffs_interp)
    {
        float3 Tha = make_float3(0.0f);
        if (true) //(!b_cam_outside && !b_light_outside)
        {


            for (unsigned int i = 0; i < iNNZ_Y; ++i)
            {
                const float l1 = tex2D<float>(lp.btf[0].tx_LNZ1, (i + 0.5f) / iNNZ_Y, patchIdx);   // fetch the spatial location 
                const float l2 = tex2D<float>(lp.btf[0].tx_LNZ2, (i + 0.5f) / iNNZ_Y, patchIdx);
                const float l3 = tex2D<float>(lp.btf[0].tx_LNZ3, (i + 0.5f) / iNNZ_Y, patchIdx);   // fetch the angular location 
                const float l4 = tex2D<float>(lp.btf[0].tx_LNZ4, (i + 0.5f) / iNNZ_Y, patchIdx);

                const float val = tex2D<float>(lp.btf[0].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_Y, patchIdx);  // fetch the non-zero value

                const float U1 = tex3D<float>(lp.btf[0].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_Y + 0.5f);   // fetch the dictionary value
                const float U2 = tex3D<float>(lp.btf[0].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_Y + 0.5f);

                const float U3_0 = tex3D<float>(lp.btf[0].tx_EnsembleDic3, light_indx[0] + 0.5f, l3 + 0.5f, iMemb_Y + 0.5f); // light
                const float U4_0 = tex3D<float>(lp.btf[0].tx_EnsembleDic4, cam_indx[0] + 0.5f, l4 + 0.5f, iMemb_Y + 0.5f); // camera

                const float U3_1 = tex3D<float>(lp.btf[0].tx_EnsembleDic3, light_indx[1] + 0.5f, l3 + 0.5f, iMemb_Y + 0.5f);
                const float U4_1 = tex3D<float>(lp.btf[0].tx_EnsembleDic4, cam_indx[1] + 0.5f, l4 + 0.5f, iMemb_Y + 0.5f);


                const float U3_2 = tex3D<float>(lp.btf[0].tx_EnsembleDic3, light_indx[2] + 0.5f, l3 + 0.5f, iMemb_Y + 0.5f);
                const float U4_2 = tex3D<float>(lp.btf[0].tx_EnsembleDic4, cam_indx[2] + 0.5f, l4 + 0.5f, iMemb_Y + 0.5f);


                // Barycentric interpolation on cameras and de-normalize the data
                /*float M0 = cam_coeffs[0] * U4_0 * std_vals[0][0] + cam_coeffs[1] * U4_1 * std_vals[0][1] + cam_coeffs[2] * U4_2 * std_vals[0][2];
                float M1 = cam_coeffs[0] * U4_0 * std_vals[1][0] + cam_coeffs[1] * U4_1 * std_vals[1][1] + cam_coeffs[2] * U4_2 * std_vals[1][2];
                float M2 = cam_coeffs[0] * U4_0 * std_vals[2][0] + cam_coeffs[1] * U4_1 * std_vals[2][1] + cam_coeffs[2] * U4_2 * std_vals[2][2];*/
                //Combine with Barycentric interpolation by changing the light
                //Tha.x += val * U1 * U2 * (light_coeffs[0] * U3_0 * M0 + light_coeffs[1] * U3_1 * M1 + light_coeffs[2] * U3_2 * M2);

                float M0 = cam_lerp[0] * U4_0 * std_vals[0][0] + cam_lerp[1] * U4_1 * std_vals[0][1];
                float M1 = cam_lerp[0] * U4_0 * std_vals[1][0] + cam_lerp[1] * U4_1 * std_vals[1][1];
                Tha.x += val * U1 * U2 * (light_lerp[0] * U3_0 * M0 + light_lerp[1] * U3_1 * M1);
                //Tha.x += val * U1 * U2 * U3_0 * U4_0;


                // Tha.x += val * U1 * U2 * (std00*light_v*U3_0 + std01*light_w* U3_1 + std02 * light_u * U3_2) * (cam_v * U4_0 + cam_w * U4_1 + cam_u * U4_2);
            }

            for (unsigned int i = 0; i < iNNZ_U; ++i)
            {
                const float l1 = tex2D<float>(lp.btf[1].tx_LNZ1, (i + 0.5f) / iNNZ_U, patchIdx);   // patchIdx * sparsity + i + 1
                const float l2 = tex2D<float>(lp.btf[1].tx_LNZ2, (i + 0.5f) / iNNZ_U, patchIdx);
                const float l3 = tex2D<float>(lp.btf[1].tx_LNZ3, (i + 0.5f) / iNNZ_U, patchIdx);
                const float l4 = tex2D<float>(lp.btf[1].tx_LNZ4, (i + 0.5f) / iNNZ_U, patchIdx);

                const float val = tex2D<float>(lp.btf[1].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_U, patchIdx);

                const float U1 = tex3D<float>(lp.btf[1].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_U + 0.5f);
                const float U2 = tex3D<float>(lp.btf[1].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_U + 0.5f);

                const float U3_0 = tex3D<float>(lp.btf[1].tx_EnsembleDic3, light_indx[0] + 0.5f, l3 + 0.5f, iMemb_U + 0.5f); // light
                const float U4_0 = tex3D<float>(lp.btf[1].tx_EnsembleDic4, cam_indx[0] + 0.5f, l4 + 0.5f, iMemb_U + 0.5f); // camera

                const float U3_1 = tex3D<float>(lp.btf[1].tx_EnsembleDic3, light_indx[1] + 0.5f, l3 + 0.5f, iMemb_U + 0.5f);
                const float U4_1 = tex3D<float>(lp.btf[1].tx_EnsembleDic4, cam_indx[1] + 0.5f, l4 + 0.5f, iMemb_U + 0.5f);


                const float U3_2 = tex3D<float>(lp.btf[1].tx_EnsembleDic3, light_indx[2] + 0.5f, l3 + 0.5f, iMemb_U + 0.5f);
                const float U4_2 = tex3D<float>(lp.btf[1].tx_EnsembleDic4, cam_indx[2] + 0.5f, l4 + 0.5f, iMemb_U + 0.5f);


                float M0 = cam_lerp[0] * U4_0 * std_vals[0][0] + cam_lerp[1] * U4_1 * std_vals[0][1];
                float M1 = cam_lerp[0] * U4_0 * std_vals[1][0] + cam_lerp[1] * U4_1 * std_vals[1][1];
                Tha.y += val * U1 * U2 * (light_lerp[0] * U3_0 * M0 + light_lerp[1] * U3_1 * M1);

            }
            for (unsigned int i = 0; i < iNNZ_V; ++i)
            {
                const float l1 = tex2D<float>(lp.btf[2].tx_LNZ1, (i + 0.5f) / iNNZ_V, patchIdx);   // patchIdx * sparsity + i + 1
                const float l2 = tex2D<float>(lp.btf[2].tx_LNZ2, (i + 0.5f) / iNNZ_V, patchIdx);
                const float l3 = tex2D<float>(lp.btf[2].tx_LNZ3, (i + 0.5f) / iNNZ_V, patchIdx);
                const float l4 = tex2D<float>(lp.btf[2].tx_LNZ4, (i + 0.5f) / iNNZ_V, patchIdx);

                const float val = tex2D<float>(lp.btf[2].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_V, patchIdx);

                const float U1 = tex3D<float>(lp.btf[2].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_V + 0.5f);
                const float U2 = tex3D<float>(lp.btf[2].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_V + 0.5f);

                const float U3_0 = tex3D<float>(lp.btf[2].tx_EnsembleDic3, light_indx[0] + 0.5f, l3 + 0.5f, iMemb_V + 0.5f); // light
                const float U4_0 = tex3D<float>(lp.btf[2].tx_EnsembleDic4, cam_indx[0] + 0.5f, l4 + 0.5f, iMemb_V + 0.5f); // camera

                const float U3_1 = tex3D<float>(lp.btf[2].tx_EnsembleDic3, light_indx[1] + 0.5f, l3 + 0.5f, iMemb_V + 0.5f);
                const float U4_1 = tex3D<float>(lp.btf[2].tx_EnsembleDic4, cam_indx[1] + 0.5f, l4 + 0.5f, iMemb_V + 0.5f);


                const float U3_2 = tex3D<float>(lp.btf[2].tx_EnsembleDic3, light_indx[2] + 0.5f, l3 + 0.5f, iMemb_V + 0.5f);
                const float U4_2 = tex3D<float>(lp.btf[2].tx_EnsembleDic4, cam_indx[2] + 0.5f, l4 + 0.5f, iMemb_V + 0.5f);

                float M0 = cam_lerp[0] * U4_0 * std_vals[0][0] + cam_lerp[1] * U4_1 * std_vals[0][1];
                float M1 = cam_lerp[0] * U4_0 * std_vals[1][0] + cam_lerp[1] * U4_1 * std_vals[1][1];
                Tha.z += val * U1 * U2 * (light_lerp[0] * U3_0 * M0 + light_lerp[1] * U3_1 * M1);
                //Tha.z += val * U1 * U2 * U3_0 * U4_0;
            }


            //Tha = Tha * std_vals[0][0] + mean_vals[0][0];
            Tha = Tha + meanvl;

            // YUV2RGB
            float3 tmp;
            tmp.x = Tha.x + 1.13983f * Tha.z;
            tmp.y = Tha.x - 0.39465f * Tha.y - 0.5806f * Tha.z;
            tmp.z = Tha.x + 2.03211f * Tha.y;

            // Inv log to get RGB values
            f_r = (expf(tmp) - 1.0e-5f);

        }
        else
        {

        }
    }


    //===================================== Loop interpolation ======================================================
    else
    {

        for (int l_idx = 0; l_idx < 2; l_idx++) //light
        {
            for (int c_idx = 0; c_idx < 2; c_idx++) // cam
            {

                float3 Tha = make_float3(0.0f);
                for (unsigned int i = 0; i < iNNZ_Y; ++i)
                {
                    const float l1 = tex2D<float>(lp.btf[0].tx_LNZ1, (i + 0.5f) / iNNZ_Y, patchIdx);   // fetch the spatial location 
                    const float l2 = tex2D<float>(lp.btf[0].tx_LNZ2, (i + 0.5f) / iNNZ_Y, patchIdx);
                    const float l3 = tex2D<float>(lp.btf[0].tx_LNZ3, (i + 0.5f) / iNNZ_Y, patchIdx);   // fetch the angular location 
                    const float l4 = tex2D<float>(lp.btf[0].tx_LNZ4, (i + 0.5f) / iNNZ_Y, patchIdx);

                    const float val = tex2D<float>(lp.btf[0].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_Y, patchIdx);  // fetch the non-zero value

                    const float U1 = tex3D<float>(lp.btf[0].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_Y + 0.5f);   // fetch the dictionary value
                    const float U2 = tex3D<float>(lp.btf[0].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_Y + 0.5f);
                    const float U3 = tex3D<float>(lp.btf[0].tx_EnsembleDic3, light_indx[l_idx] + 0.5f, l3 + 0.5f, iMemb_Y + 0.5f);
                    const float U4 = tex3D<float>(lp.btf[0].tx_EnsembleDic4, cam_indx[c_idx] + 0.5f, l4 + 0.5f, iMemb_Y + 0.5f);

                    Tha.x += val * U1 * U2 * U3 * U4;
                }
                for (unsigned int i = 0; i < iNNZ_U; ++i)
                {
                    const float l1 = tex2D<float>(lp.btf[1].tx_LNZ1, (i + 0.5f) / iNNZ_U, patchIdx);   // patchIdx * sparsity + i + 1
                    const float l2 = tex2D<float>(lp.btf[1].tx_LNZ2, (i + 0.5f) / iNNZ_U, patchIdx);
                    const float l3 = tex2D<float>(lp.btf[1].tx_LNZ3, (i + 0.5f) / iNNZ_U, patchIdx);
                    const float l4 = tex2D<float>(lp.btf[1].tx_LNZ4, (i + 0.5f) / iNNZ_U, patchIdx);

                    const float val = tex2D<float>(lp.btf[1].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_U, patchIdx);

                    const float U1 = tex3D<float>(lp.btf[1].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_U + 0.5f);
                    const float U2 = tex3D<float>(lp.btf[1].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_U + 0.5f);
                    const float U3 = tex3D<float>(lp.btf[1].tx_EnsembleDic3, light_indx[l_idx] + 0.5f, l3 + 0.5f, iMemb_U + 0.5f);
                    const float U4 = tex3D<float>(lp.btf[1].tx_EnsembleDic4, cam_indx[c_idx] + 0.5f, l4 + 0.5f, iMemb_U + 0.5f);

                    Tha.y += val * U1 * U2 * U3 * U4;
                }
                for (unsigned int i = 0; i < iNNZ_V; ++i)
                {
                    const float l1 = tex2D<float>(lp.btf[2].tx_LNZ1, (i + 0.5f) / iNNZ_V, patchIdx);   // patchIdx * sparsity + i + 1
                    const float l2 = tex2D<float>(lp.btf[2].tx_LNZ2, (i + 0.5f) / iNNZ_V, patchIdx);
                    const float l3 = tex2D<float>(lp.btf[2].tx_LNZ3, (i + 0.5f) / iNNZ_V, patchIdx);
                    const float l4 = tex2D<float>(lp.btf[2].tx_LNZ4, (i + 0.5f) / iNNZ_V, patchIdx);

                    const float val = tex2D<float>(lp.btf[2].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_V, patchIdx);

                    const float U1 = tex3D<float>(lp.btf[2].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_V + 0.5f);
                    const float U2 = tex3D<float>(lp.btf[2].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_V + 0.5f);
                    const float U3 = tex3D<float>(lp.btf[2].tx_EnsembleDic3, light_indx[l_idx] + 0.5f, l3 + 0.5f, iMemb_V + 0.5f);
                    const float U4 = tex3D<float>(lp.btf[2].tx_EnsembleDic4, cam_indx[c_idx] + 0.5f, l4 + 0.5f, iMemb_V + 0.5f);

                    Tha.z += val * U1 * U2 * U3 * U4;
                }

                Tha = Tha * std_vals[l_idx][c_idx] + mean_vals[l_idx][c_idx];   // de-normalize the data

                // YUV2RGB
                float3 tmp;
                tmp.x = Tha.x + 1.13983 * Tha.z;
                tmp.y = Tha.x - 0.39465 * Tha.y - 0.5806 * Tha.z;
                tmp.z = Tha.x + 2.03211 * Tha.y;

                // Inv log to get RGB values
                float3 RGB_vals = (expf(tmp) - 1.0e-5f);
                colval += cam_lerp[c_idx] * light_lerp[l_idx] * RGB_vals;//

                //totalW += cam_coeffs[j] * light_coeffs[k];


            }
        }

        f_r = colval;
    }
    float3 result = f_r * E;

#ifdef PASS_PAYLOAD_POINTER
    PayloadRadiance* prd = getPayload();
    prd->result = result;
#else
    setPayloadResult(result);
#endif
}

// without interpolation in coefficients space (slower space)
// with different kernel functions (smoother results)
extern "C" __global__ void __closesthit__btf_filtering()
{
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;

  const LaunchParams& lp = launch_params;
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
  if(depth > lp.max_depth)
  {
    prd->result = make_float3(0.0f);
    return;
  }
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
  if(depth > 0)
    return;
#endif

  // Retrieve hit info
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  const float3& x = geom.P;
  const float3 wo = -optixGetWorldRayDirection(); // -normalize(lp.W); // Use lp.W to have only one direction toward the observer
  const float2 texcoords = geom.UV*lp.tex_scale;
  float3 n = normalize(geom.N);
  float cos_theta_o = dot(n, wo);
  const float s = copysignf(1.0f, cos_theta_o);
  cos_theta_o *= s;
  n *= s;

  float3 E = make_float3(0.0f), wi;
  const float no_of_sources = static_cast<float>(lp.lights.count);
  const float bgc = length(lp.miss_color);
  const float prob = bgc/(no_of_sources + bgc);
  const float xi = rnd(t);
  if(xi < prob)
  {
    // Environment map importance sampling
    //*
    sample_environment(x, wi, E, t);
    const float cos_theta_i = dot(wi, n);
    if(cos_theta_i <= 0.0f)
      return;
    if(traceOcclusion(lp.handle, x, wi, tmin, tmax))
      return;
    E *= cos_theta_i/prob;
  }
  else
  {
    // Direct illumination
    const unsigned int idx = static_cast<unsigned int>(rnd(t)*no_of_sources);
    const Directional& light = lp.lights[idx];
    wi = -light.direction;
    const float cos_theta_i = dot(wi, n);
    if(dot(wi, n) <= 0.0f)
      return;
    // Shadow ray
    if(traceOcclusion(lp.handle, x, wi, tmin, tmax))
      return;
    E = light.emission*cos_theta_i*(no_of_sources + bgc);
  }

  // Texture wrapping (?)
  const float tex_s = texcoords.x; // geom.UV.x - floorf(geom.UV.x);
  const float tex_t = texcoords.y; // -geom.UV.y - floorf(-geom.UV.y);
  //const float tex_s = geom.UV.x*lp.tex_scale - 0.001f*(lp.tex_scale - 1.0f);
  //const float tex_t = (1.0f - geom.UV.y)*lp.tex_scale - 0.001f*(lp.tex_scale - 1.0f);

  // Change of basis to tangent space
  float3 ta, bi;
  onb_consistent(n, ta, bi);
  const float3 b1 = make_float3(ta.x, bi.x, n.x);
  const float3 b2 = make_float3(ta.y, bi.y, n.y);
  const float3 b3 = make_float3(ta.z, bi.z, n.z);
  const float3 t_wi = b1*wi.x + b2*wi.y + b3*wi.z;
  const float3 t_wo = b1*wo.x + b2*wo.y + b3*wo.z;

  const uint2 spatialPos = make_uint2(tex_s * 400, tex_t * 400); //each image in BTF is of size 400x400
  const uint2 patchLoc = make_uint2(spatialPos.x / 10, spatialPos.y / 10);
  const float2 spatialLoc = make_float2(spatialPos.x % 10, spatialPos.y % 10);
  unsigned int patchId = patchLoc.y * 40 + patchLoc.x;
  float patchIdx = static_cast<float>(patchId + 0.5f) / 1600.0f;  // normalize the index to use the wrap address mode (to be able to repeat BTF)

  float iNNZ_Y = tex1D<float>(lp.btf[0].tx_NNZ, patchIdx);
  float iMemb_Y = tex1D<float>(lp.btf[0].tx_Memb, patchIdx);

  float iNNZ_U = tex1D<float>(lp.btf[1].tx_NNZ, patchIdx);
  float iMemb_U = tex1D<float>(lp.btf[1].tx_Memb, patchIdx);

  float iNNZ_V = tex1D<float>(lp.btf[2].tx_NNZ, patchIdx);
  float iMemb_V = tex1D<float>(lp.btf[2].tx_Memb, patchIdx);

  // Init look-up variables
  float2 ang_coords[3];
  float3 dist_cams = make_float3(tmax);
  float3 dist_lights = make_float3(tmax);

  // Find the closest camera and light solid angle bins
  for(unsigned int i = 0; i < 151; ++i)
  {
    const float2 angle_list = make_float2(lp.phi[i], lp.theta[i]);
    const float3 bin_dir = spherical_direction(cosf(angle_list.y), sinf(angle_list.y), angle_list.x);
    const float cam_dist = length(t_wo - bin_dir);
    const float light_dist = length(t_wi - bin_dir);

    if(cam_dist < dist_cams.x)
    {
      dist_cams.z = dist_cams.y;
      ang_coords[2].y = ang_coords[1].y;
      dist_cams.y = dist_cams.x;
      ang_coords[1].y = ang_coords[0].y;
      dist_cams.x = cam_dist;
      ang_coords[0].y = static_cast<float>(i);
    }
    else if(cam_dist < dist_cams.y)
    {
      dist_cams.z = dist_cams.y;
      ang_coords[2].y = ang_coords[1].y;
      dist_cams.y = cam_dist;
      ang_coords[1].y = static_cast<float>(i);
    }
    else if(cam_dist < dist_cams.z)
    {
      dist_cams.z = cam_dist;
      ang_coords[2].y = static_cast<float>(i);
    }

    if(light_dist < dist_lights.x)
    {
      dist_lights.z = dist_lights.y;
      ang_coords[2].x = ang_coords[1].x;
      dist_lights.y = dist_lights.x;
      ang_coords[1].x = ang_coords[0].x;
      dist_lights.x = light_dist;
      ang_coords[0].x = static_cast<float>(i);
    }
    else if(light_dist < dist_lights.y)
    {
      dist_lights.z = dist_lights.y;
      ang_coords[2].x = ang_coords[1].x;
      dist_lights.y = light_dist;
      ang_coords[1].x = static_cast<float>(i);
    }
    else if(light_dist < dist_lights.z)
    {
      dist_lights.z = light_dist;
      ang_coords[2].x = static_cast<float>(i);
    }
  }

  // How to fetch the elements of adjancy map
  //const float m = tex3D<float>(lp.adj_map, x + 0.5f, y + 0.5f, z + 0.5f);
  //printf("%f\n", m);

  float3 temp[8];
  for(unsigned int j = 0; j < 8; ++j)
  {
    const unsigned int j_x = j / 3;
    const unsigned int j_y = j % 3;
    const float2 angularLoc = make_float2(ang_coords[j_x].x, ang_coords[j_y].y);
    float3 Tha = make_float3(0.0f);

    for(unsigned int i = 0; i < iNNZ_Y; ++i)
    {
      const float l1 = tex2D<float>(lp.btf[0].tx_LNZ1, (i + 0.5f) / iNNZ_Y, patchIdx);   // fetch the spatial location 
      const float l2 = tex2D<float>(lp.btf[0].tx_LNZ2, (i + 0.5f) / iNNZ_Y, patchIdx);
      const float l3 = tex2D<float>(lp.btf[0].tx_LNZ3, (i + 0.5f) / iNNZ_Y, patchIdx);   // fetch the angular location 
      const float l4 = tex2D<float>(lp.btf[0].tx_LNZ4, (i + 0.5f) / iNNZ_Y, patchIdx);

      const float val = tex2D<float>(lp.btf[0].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_Y, patchIdx);  // fetch the non-zero value

      const float U1 = tex3D<float>(lp.btf[0].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_Y + 0.5f);   // fetch the dictionary value
      const float U2 = tex3D<float>(lp.btf[0].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_Y + 0.5f);
      const float U3 = tex3D<float>(lp.btf[0].tx_EnsembleDic3, angularLoc.x + 0.5f, l3 + 0.5f, iMemb_Y + 0.5f);
      const float U4 = tex3D<float>(lp.btf[0].tx_EnsembleDic4, angularLoc.y + 0.5f, l4 + 0.5f, iMemb_Y + 0.5f);

      Tha.x += val * U1 * U2 * U3 * U4;
    }
    for(unsigned int i = 0; i < iNNZ_U; ++i)
    {
      const float l1 = tex2D<float>(lp.btf[1].tx_LNZ1, (i + 0.5f) / iNNZ_U, patchIdx);   // patchIdx * sparsity + i + 1
      const float l2 = tex2D<float>(lp.btf[1].tx_LNZ2, (i + 0.5f) / iNNZ_U, patchIdx);
      const float l3 = tex2D<float>(lp.btf[1].tx_LNZ3, (i + 0.5f) / iNNZ_U, patchIdx);
      const float l4 = tex2D<float>(lp.btf[1].tx_LNZ4, (i + 0.5f) / iNNZ_U, patchIdx);

      const float val = tex2D<float>(lp.btf[1].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_U, patchIdx);

      const float U1 = tex3D<float>(lp.btf[1].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_U + 0.5f);
      const float U2 = tex3D<float>(lp.btf[1].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_U + 0.5f);
      const float U3 = tex3D<float>(lp.btf[1].tx_EnsembleDic3, angularLoc.x + 0.5f, l3 + 0.5f, iMemb_U + 0.5f);
      const float U4 = tex3D<float>(lp.btf[1].tx_EnsembleDic4, angularLoc.y + 0.5f, l4 + 0.5f, iMemb_U + 0.5f);

      Tha.y += val * U1 * U2 * U3 * U4;
    }
    for(unsigned int i = 0; i < iNNZ_V; ++i)
    {
      const float l1 = tex2D<float>(lp.btf[2].tx_LNZ1, (i + 0.5f) / iNNZ_V, patchIdx);   // patchIdx * sparsity + i + 1
      const float l2 = tex2D<float>(lp.btf[2].tx_LNZ2, (i + 0.5f) / iNNZ_V, patchIdx);
      const float l3 = tex2D<float>(lp.btf[2].tx_LNZ3, (i + 0.5f) / iNNZ_V, patchIdx);
      const float l4 = tex2D<float>(lp.btf[2].tx_LNZ4, (i + 0.5f) / iNNZ_V, patchIdx);

      const float val = tex2D<float>(lp.btf[2].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_V, patchIdx);

      const float U1 = tex3D<float>(lp.btf[2].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_V + 0.5f);
      const float U2 = tex3D<float>(lp.btf[2].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_V + 0.5f);
      const float U3 = tex3D<float>(lp.btf[2].tx_EnsembleDic3, angularLoc.x + 0.5f, l3 + 0.5f, iMemb_V + 0.5f);
      const float U4 = tex3D<float>(lp.btf[2].tx_EnsembleDic4, angularLoc.y + 0.5f, l4 + 0.5f, iMemb_V + 0.5f);

      Tha.z += val * U1 * U2 * U3 * U4;
    }
    unsigned int idx = angularLoc.y * 151 + angularLoc.x;
    Tha = Tha * lp.std[idx] + lp.mean[idx];   // de-normalize the data

    // YUV2RGB
    float3 tmp;
    tmp.x = Tha.x + 1.13983f * Tha.z;
    tmp.y = Tha.x - 0.39465f * Tha.y - 0.5806f * Tha.z;
    tmp.z = Tha.x + 2.03211f * Tha.y;

    // Inv log to get RGB values
    temp[j] = (expf(tmp) - 1.0e-5f);
  }
//#define NEAREST
//#define BILINEAR
//#define TRIANGULAR
//#define EPANECHNIKOV
//#define BIWEIGHT
//#define CUBIC
#define GAUSSIAN 20.0f

#ifdef NEAREST
  const float3& f_r = temp[0];
#elif defined(BILINEAR)
  const float a = 0.5f + 0.5f * (dist_cams.y - dist_cams.x) / dist_cams.y;
  const float b = 0.5f + 0.5f * (dist_lights.y - dist_lights.x) / dist_lights.y;
  const float3 f_r = lerp(lerp(temp[4], temp[3], a), lerp(temp[1], temp[0], a), b);
#else
  const float3 sqr_dist_lights = dist_lights*dist_lights;
  const float3 sqr_dist_cams = dist_cams*dist_cams;
  const float denom = sqr_dist_lights.z + sqr_dist_cams.z;
#ifdef TRIANGULAR
  const float w0 = 1.0f - sqrtf((sqr_dist_lights.x + sqr_dist_cams.x)/denom);
  const float w1 = 1.0f - sqrtf((sqr_dist_lights.x + sqr_dist_cams.y)/denom);
  const float w2 = 1.0f - sqrtf((sqr_dist_lights.x + sqr_dist_cams.z)/denom);
  const float w3 = 1.0f - sqrtf((sqr_dist_lights.y + sqr_dist_cams.x)/denom);
  const float w4 = 1.0f - sqrtf((sqr_dist_lights.y + sqr_dist_cams.y)/denom);
  const float w5 = 1.0f - sqrtf((sqr_dist_lights.y + sqr_dist_cams.z)/denom);
  const float w6 = 1.0f - sqrtf((sqr_dist_lights.z + sqr_dist_cams.x)/denom);
  const float w7 = 1.0f - sqrtf((sqr_dist_lights.z + sqr_dist_cams.y)/denom);
#elif defined(EPANECHNIKOV)
  const float w0 = 1.0f - (sqr_dist_lights.x + sqr_dist_cams.x)/denom;
  const float w1 = 1.0f - (sqr_dist_lights.x + sqr_dist_cams.y)/denom;
  const float w2 = 1.0f - (sqr_dist_lights.x + sqr_dist_cams.z)/denom;
  const float w3 = 1.0f - (sqr_dist_lights.y + sqr_dist_cams.x)/denom;
  const float w4 = 1.0f - (sqr_dist_lights.y + sqr_dist_cams.y)/denom;
  const float w5 = 1.0f - (sqr_dist_lights.y + sqr_dist_cams.z)/denom;
  const float w6 = 1.0f - (sqr_dist_lights.z + sqr_dist_cams.x)/denom;
  const float w7 = 1.0f - (sqr_dist_lights.z + sqr_dist_cams.y)/denom;
#elif defined(BIWEIGHT)
  const float w0 = powf(1.0f - (sqr_dist_lights.x + sqr_dist_cams.x)/denom, 2.0f);
  const float w1 = powf(1.0f - (sqr_dist_lights.x + sqr_dist_cams.y)/denom, 2.0f);
  const float w2 = powf(1.0f - (sqr_dist_lights.x + sqr_dist_cams.z)/denom, 2.0f);
  const float w3 = powf(1.0f - (sqr_dist_lights.y + sqr_dist_cams.x)/denom, 2.0f);
  const float w4 = powf(1.0f - (sqr_dist_lights.y + sqr_dist_cams.y)/denom, 2.0f);
  const float w5 = powf(1.0f - (sqr_dist_lights.y + sqr_dist_cams.z)/denom, 2.0f);
  const float w6 = powf(1.0f - (sqr_dist_lights.z + sqr_dist_cams.x)/denom, 2.0f);
  const float w7 = powf(1.0f - (sqr_dist_lights.z + sqr_dist_cams.y)/denom, 2.0f);
#elif defined(CUBIC)
  const float w0 = powf(1.0f - (sqr_dist_lights.x + sqr_dist_cams.x)/denom, 3.0f);
  const float w1 = powf(1.0f - (sqr_dist_lights.x + sqr_dist_cams.y)/denom, 3.0f);
  const float w2 = powf(1.0f - (sqr_dist_lights.x + sqr_dist_cams.z)/denom, 3.0f);
  const float w3 = powf(1.0f - (sqr_dist_lights.y + sqr_dist_cams.x)/denom, 3.0f);
  const float w4 = powf(1.0f - (sqr_dist_lights.y + sqr_dist_cams.y)/denom, 3.0f);
  const float w5 = powf(1.0f - (sqr_dist_lights.y + sqr_dist_cams.z)/denom, 3.0f);
  const float w6 = powf(1.0f - (sqr_dist_lights.z + sqr_dist_cams.x)/denom, 3.0f);
  const float w7 = powf(1.0f - (sqr_dist_lights.z + sqr_dist_cams.y)/denom, 3.0f);
#else
  const float beta = GAUSSIAN;
  const float w = 1.0f - expf(-beta);
  const float w0 = 1.0f - (1.0f - expf(-beta*(sqr_dist_lights.x + sqr_dist_cams.x)/(2.0f*denom)))/w;
  const float w1 = 1.0f - (1.0f - expf(-beta*(sqr_dist_lights.x + sqr_dist_cams.y)/(2.0f*denom)))/w;
  const float w2 = 1.0f - (1.0f - expf(-beta*(sqr_dist_lights.x + sqr_dist_cams.z)/(2.0f*denom)))/w;
  const float w3 = 1.0f - (1.0f - expf(-beta*(sqr_dist_lights.y + sqr_dist_cams.x)/(2.0f*denom)))/w;
  const float w4 = 1.0f - (1.0f - expf(-beta*(sqr_dist_lights.y + sqr_dist_cams.y)/(2.0f*denom)))/w;
  const float w5 = 1.0f - (1.0f - expf(-beta*(sqr_dist_lights.y + sqr_dist_cams.z)/(2.0f*denom)))/w;
  const float w6 = 1.0f - (1.0f - expf(-beta*(sqr_dist_lights.z + sqr_dist_cams.x)/(2.0f*denom)))/w;
  const float w7 = 1.0f - (1.0f - expf(-beta*(sqr_dist_lights.z + sqr_dist_cams.y)/(2.0f*denom)))/w;
#endif
  // blend
  const float3 f_r = (temp[0]*w0 + temp[1]*w1 + temp[2]*w2 + temp[3]*w3 + temp[4]*w4 + temp[5]*w5 + temp[6]*w6 + temp[7]*w7)/(w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7);
#endif
  const float3 result = f_r * E;

#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}

// without interpolation in coefficients space (slower version)
extern "C" __global__ void __closesthit__btf_simple()
{
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;

  const LaunchParams& lp = launch_params;
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
  if(depth > lp.max_depth)
  {
    prd->result = make_float3(0.0f);
    return;
  }
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
  if(depth > 0)
    return;
#endif

  // Retrieve hit info
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  const float3& x = geom.P;
  const float3 wo = -optixGetWorldRayDirection(); // -normalize(lp.W); // Use lp.W to have only one direction toward the observer
  const float2 texcoords = geom.UV*lp.tex_scale;
  float3 n = normalize(geom.N);
  float cos_theta_o = dot(n, wo);
  const float s = copysignf(1.0f, cos_theta_o);
  cos_theta_o *= s;
  n *= s;

  float3 E = make_float3(0.0f), wi;
  const float no_of_sources = static_cast<float>(lp.lights.count);
  const float bgc = length(lp.miss_color);
  const float prob = bgc/(no_of_sources + bgc);
  const float xi = rnd(t);
  if(xi < prob)
  {
    // Environment map importance sampling
    //*
    sample_environment(x, wi, E, t);
    const float cos_theta_i = dot(wi, n);
    if(cos_theta_i <= 0.0f)
      return;
    if(traceOcclusion(lp.handle, x, wi, tmin, tmax))
      return;
    E *= cos_theta_i/prob;
  }
  else
  {
    // Direct illumination
    const unsigned int idx = static_cast<unsigned int>(rnd(t)*no_of_sources);
    const Directional& light = lp.lights[idx];
    wi = -light.direction;
    const float cos_theta_i = dot(wi, n);
    if(dot(wi, n) <= 0.0f)
      return;
    // Shadow ray
    if(traceOcclusion(lp.handle, x, wi, tmin, tmax))
      return;
    E = light.emission*cos_theta_i*(no_of_sources + bgc);
  }

  // Texture wrapping 
  const float tex_s = texcoords.x; // geom.UV.x - floorf(geom.UV.x);
  const float tex_t = texcoords.y; // -geom.UV.y - floorf(-geom.UV.y);

  // Change of basis to tangent space
  float3 ta, bi;
  onb_consistent(n, ta, bi);
  const float3 b1 = make_float3(ta.x, bi.x, n.x);
  const float3 b2 = make_float3(ta.y, bi.y, n.y);
  const float3 b3 = make_float3(ta.z, bi.z, n.z);
  const float3 t_wi = b1*wi.x + b2*wi.y + b3*wi.z;
  const float3 t_wo = b1*wo.x + b2*wo.y + b3*wo.z;

  const uint2 spatialPos = make_uint2(tex_s * 400, tex_t * 400); //each image in BTF is of size 400x400
  const uint2 patchLoc = make_uint2(spatialPos.x / 10, spatialPos.y / 10);
  const float2 spatialLoc = make_float2(spatialPos.x % 10, spatialPos.y % 10);
  unsigned int patchId = patchLoc.y * 40 + patchLoc.x;
  float patchIdx = static_cast<float>(patchId + 0.5f) / 1600.0f;  // normalize the index to use the wrap address mode (to be able to repeat BTF)

  float iNNZ_Y = tex1D<float>(lp.btf[0].tx_NNZ, patchIdx);
  float iMemb_Y = tex1D<float>(lp.btf[0].tx_Memb, patchIdx);

  float iNNZ_U = tex1D<float>(lp.btf[1].tx_NNZ, patchIdx);
  float iMemb_U = tex1D<float>(lp.btf[1].tx_Memb, patchIdx);

  float iNNZ_V = tex1D<float>(lp.btf[2].tx_NNZ, patchIdx);
  float iMemb_V = tex1D<float>(lp.btf[2].tx_Memb, patchIdx);

  // Init look-up variables
  float2 ang_coords[2];
  float first_point_cam = tmax;
  float second_point_cam = tmax;
  float first_point_li = tmax;
  float second_point_li = tmax;
  
  // Find the closest camera and light solid angle bins
  for (unsigned int i = 0; i < 151; ++i)
  {
  	const float2 angle_list = make_float2(lp.phi[i], lp.theta[i]);
  	const float3 bin_dir = spherical_direction(cosf(angle_list.y), sinf(angle_list.y), angle_list.x);
  	const float cam_dist = length(t_wo - bin_dir);
  	const float light_dist = length(t_wi - bin_dir);

  	if (cam_dist < first_point_cam)
  	{
  		second_point_cam = first_point_cam;
  		ang_coords[1].y = ang_coords[0].y;
  		first_point_cam = cam_dist;
  		ang_coords[0].y = static_cast<float>(i);
  	}
  	else if (cam_dist < second_point_cam)
  	{
  		second_point_cam = cam_dist;
  		ang_coords[1].y = static_cast<float>(i);
  	}
  
  	if (light_dist < first_point_li)
  	{
  		second_point_li = first_point_li;
  		ang_coords[1].x = ang_coords[0].x;
  		first_point_li = light_dist;
  		ang_coords[0].x = static_cast<float>(i);
  	}
  	else if (light_dist < second_point_li)
  	{
  		second_point_li = light_dist;
  		ang_coords[1].x = static_cast<float>(i);
  	}
  }

  float3 temp[4];
  for (unsigned int j = 0; j < 4; ++j)
  {
  	const unsigned int j_x = j / 2;
  	const unsigned int j_y = j % 2;
  	const float2 angularLoc = make_float2(ang_coords[j_x].x, ang_coords[j_y].y);
  	float3 Tha = make_float3(0.0f);

  	for (unsigned int i = 0; i < iNNZ_Y; ++i)
  	{
  		const float l1 = tex2D<float>(lp.btf[0].tx_LNZ1, (i + 0.5f) / iNNZ_Y, patchIdx);   // fetch the spatial location 
  		const float l2 = tex2D<float>(lp.btf[0].tx_LNZ2, (i + 0.5f) / iNNZ_Y, patchIdx);
  		const float l3 = tex2D<float>(lp.btf[0].tx_LNZ3, (i + 0.5f) / iNNZ_Y, patchIdx);   // fetch the angular location 
  		const float l4 = tex2D<float>(lp.btf[0].tx_LNZ4, (i + 0.5f) / iNNZ_Y, patchIdx);

  		const float val = tex2D<float>(lp.btf[0].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_Y, patchIdx);  // fetch the non-zero value

  		const float U1 = tex3D<float>(lp.btf[0].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_Y + 0.5f);   // fetch the dictionary value
  		const float U2 = tex3D<float>(lp.btf[0].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_Y + 0.5f);
  		const float U3 = tex3D<float>(lp.btf[0].tx_EnsembleDic3, angularLoc.x + 0.5f, l3 + 0.5f, iMemb_Y + 0.5f);
  		const float U4 = tex3D<float>(lp.btf[0].tx_EnsembleDic4, angularLoc.y + 0.5f, l4 + 0.5f, iMemb_Y + 0.5f);

  		Tha.x += val * U1 * U2 * U3 * U4;
  	}
  	for (unsigned int i = 0; i < iNNZ_U; ++i)
  	{
  		const float l1 = tex2D<float>(lp.btf[1].tx_LNZ1, (i + 0.5f) / iNNZ_U, patchIdx);   // patchIdx * sparsity + i + 1
  		const float l2 = tex2D<float>(lp.btf[1].tx_LNZ2, (i + 0.5f) / iNNZ_U, patchIdx);
  		const float l3 = tex2D<float>(lp.btf[1].tx_LNZ3, (i + 0.5f) / iNNZ_U, patchIdx);
  		const float l4 = tex2D<float>(lp.btf[1].tx_LNZ4, (i + 0.5f) / iNNZ_U, patchIdx);

  		const float val = tex2D<float>(lp.btf[1].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_U, patchIdx);

  		const float U1 = tex3D<float>(lp.btf[1].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_U + 0.5f);
  		const float U2 = tex3D<float>(lp.btf[1].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_U + 0.5f);
  		const float U3 = tex3D<float>(lp.btf[1].tx_EnsembleDic3, angularLoc.x + 0.5f, l3 + 0.5f, iMemb_U + 0.5f);
  		const float U4 = tex3D<float>(lp.btf[1].tx_EnsembleDic4, angularLoc.y + 0.5f, l4 + 0.5f, iMemb_U + 0.5f);

  		Tha.y += val * U1 * U2 * U3 * U4;
  	}
  	for (unsigned int i = 0; i < iNNZ_V; ++i)
  	{
  		const float l1 = tex2D<float>(lp.btf[2].tx_LNZ1, (i + 0.5f) / iNNZ_V, patchIdx);   // patchIdx * sparsity + i + 1
  		const float l2 = tex2D<float>(lp.btf[2].tx_LNZ2, (i + 0.5f) / iNNZ_V, patchIdx);
  		const float l3 = tex2D<float>(lp.btf[2].tx_LNZ3, (i + 0.5f) / iNNZ_V, patchIdx);
  		const float l4 = tex2D<float>(lp.btf[2].tx_LNZ4, (i + 0.5f) / iNNZ_V, patchIdx);

  		const float val = tex2D<float>(lp.btf[2].tx_NZ2DCoeffvar, (i + 0.5f) / iNNZ_V, patchIdx);

  		const float U1 = tex3D<float>(lp.btf[2].tx_EnsembleDic1, spatialLoc.x + 0.5f, l1 + 0.5f, iMemb_V + 0.5f);
  		const float U2 = tex3D<float>(lp.btf[2].tx_EnsembleDic2, spatialLoc.y + 0.5f, l2 + 0.5f, iMemb_V + 0.5f);
  		const float U3 = tex3D<float>(lp.btf[2].tx_EnsembleDic3, angularLoc.x + 0.5f, l3 + 0.5f, iMemb_V + 0.5f);
  		const float U4 = tex3D<float>(lp.btf[2].tx_EnsembleDic4, angularLoc.y + 0.5f, l4 + 0.5f, iMemb_V + 0.5f);

  		Tha.z += val * U1 * U2 * U3 * U4;
  	}
  	unsigned int idx = angularLoc.y * 151 + angularLoc.x;
  	Tha = Tha * lp.std[idx] + lp.mean[idx];   // de-normalize the data

  	// YUV2RGB
  	float3 tmp;
  	tmp.x = Tha.x + 1.13983f * Tha.z;
  	tmp.y = Tha.x - 0.39465f * Tha.y - 0.5806f * Tha.z;
  	tmp.z = Tha.x + 2.03211f * Tha.y;

  	// Inv log to get RGB values
  	temp[j] = (expf(tmp) - 1.0e-5f);
  }

//#define NEAREST
//#define BILINEAR
//#define TRIANGULAR
//#define EPANECHNIKOV
//#define BIWEIGHT
//#define CUBIC
#define GAUSSIAN 20.0f

#ifdef NEAREST
  const float3& f_r = temp[0];
#elif defined(BILINEAR)
  //const float a = 0.5f + 0.5f * powf((second_point_cam - first_point_cam) / second_point_cam, 0.5f);
  //const float b = 0.5f + 0.5f * powf((second_point_li - first_point_li) / second_point_li, 0.5f);
  const float a = 0.5f + 0.5f * (second_point_cam - first_point_cam) / second_point_cam;
  const float b = 0.5f + 0.5f * (second_point_li - first_point_li) / second_point_li;
  const float3 f_r = lerp(lerp(temp[3], temp[2], a), lerp(temp[1], temp[0], a), b);
#else
  const float4 dists = make_float4(first_point_li, second_point_li, first_point_cam, second_point_cam);
  const float4 sqr_dists = dists*dists;
  const float denom = sqr_dists.y + sqr_dists.w;
#ifdef TRIANGULAR
  const float w0 = 1.0f - sqrtf((sqr_dists.x + sqr_dists.z)/denom);
  const float w1 = 1.0f - sqrtf((sqr_dists.x + sqr_dists.w)/denom);
  const float w2 = 1.0f - sqrtf((sqr_dists.y + sqr_dists.z)/denom);
#elif defined(EPANECHNIKOV)
  const float w0 = 1.0f - (sqr_dists.x + sqr_dists.z)/denom;
  const float w1 = 1.0f - (sqr_dists.x + sqr_dists.w)/denom;
  const float w2 = 1.0f - (sqr_dists.y + sqr_dists.z)/denom;
#elif defined(BIWEIGHT)
  const float w0 = powf(1.0f - (sqr_dists.x + sqr_dists.z)/denom, 2.0f);
  const float w1 = powf(1.0f - (sqr_dists.x + sqr_dists.w)/denom, 2.0f);
  const float w2 = powf(1.0f - (sqr_dists.y + sqr_dists.z)/denom, 2.0f);
#elif defined(CUBIC)
  const float w0 = powf(1.0f - (sqr_dists.x + sqr_dists.z)/denom, 3.0f);
  const float w1 = powf(1.0f - (sqr_dists.x + sqr_dists.w)/denom, 3.0f);
  const float w2 = powf(1.0f - (sqr_dists.y + sqr_dists.z)/denom, 3.0f);
#else
  const float beta = GAUSSIAN;
  const float w = 1.0f - expf(-beta);
  const float w0 = 1.0f - (1.0f - expf(-beta*(sqr_dists.x + sqr_dists.z)/(2.0f*denom)))/w;
  const float w1 = 1.0f - (1.0f - expf(-beta*(sqr_dists.x + sqr_dists.w)/(2.0f*denom)))/w;
  const float w2 = 1.0f - (1.0f - expf(-beta*(sqr_dists.y + sqr_dists.z)/(2.0f*denom)))/w;
#endif
  // blend
  const float3 f_r = (temp[0]*w0 + temp[1]*w1 + temp[2]*w2)/(w0 + w1 + w2);
#endif
  const float3 result = f_r * E;

#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}
