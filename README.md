# SparseBTF

If you use this code, we kindly request that you cite the following paper:

Kavoosighafi, B., Frisvad, J. R., Hajisharif, S., Unger, J., and Miandji, E. SparseBTF: Sparse Representation Learning for Real-time Rendering of Bidirectional Texture Functions. In EGSR. 2023.

The repository contains the code for sparse representation of Bidirectional Texture Functions (BTFs), including dictionaries and two examples of sparse coefficients for SparseBTF (32). Additionally, it includes the shader code in NVIDIA OptiX SDK 7.4. 

To compile the code, you will need the following libraries:

- Eigen 3 (header-only)
- boost
- matio (https://sourceforge.net/projects/matio/)
- HDF5 (required by matio)

We have provided a solution file for Visual C++ 2022, as well as a CMake file. The code has been successfully compiled and tested under Windows 10 and Ubuntu 22.04. However, it has not been tested on MacOS.

Please note that the repository does not include the model training code. However, we have included a trained ensemble named "DictEnsOrth4D.mat" located at "project-root/bin/DataBTF/".

The UBO2014 BTFs in MATLAB's .mat file format can be downloaded from [here](https://drive.google.com/drive/folders/13guJ4x4VjN_xH6_dB-pxr5lmLZGaqQ2h).

If you have any questions or need assistance, please feel free to reach out to me at behnaz.kavoosighafi@liu.se.
