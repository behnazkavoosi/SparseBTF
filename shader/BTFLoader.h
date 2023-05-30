#pragma once

#include <string>
#include <vector>

//-----------------------------------------------------------------------------
//
// BTFLoader class declaration 
//
//-----------------------------------------------------------------------------

class BTFLoader
{
public:
  BTFLoader() : loaded_means_angles(false) { }

  void load_btf_coeff_files(const std::string& filename, unsigned int sparsity, unsigned int channel);
  void load_btf_dict_files(const std::string& filename, unsigned int num_dict, unsigned int channel);
  void load_btf_mean_std(const std::string& filename) const;
  void load_btf_angles_list(const std::string& filename) const;
  void load_btf_loc_list(const std::string& filename) const;
  void load_btf_adj_map(const std::string& filename);
  void cleanup();

protected:
  std::vector<cudaTextureObject_t> m_samplers;
  bool loaded_means_angles;
};
