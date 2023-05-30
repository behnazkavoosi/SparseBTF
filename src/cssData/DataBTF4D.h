#pragma once

#include "Data.h"
#include "../cssUtil/VclMatlab.h"
#include "../cssUtil/Patchifier.h"
#include "../cssUtil/LATools.h"
#include "../cssUtil/ImgIO.h"
#include "../cssUtil/ImgQlty.h"



template <typename T>
class CDataBTF4D : public CData<T, 4>
{

public:

	CDataBTF4D(uint32_t m_iPhi, uint32_t m_iTheta);
	virtual ~CDataBTF4D();

	virtual size_t GetNumDataElems() const;
	virtual void GetDataElemDim(std::vector<size_t>& vDims) const;
	virtual void SetColorCh(int iChannel);

	template <typename U> void ResizeLike(const CData<U, 4>* pData);
	template <typename U> void CopyPropsFrom(const CData<U, 4>* pData);

	virtual bool Init();
	virtual bool LoadFromDisk(const std::string& strDataFolder);
	virtual bool PrepareData(const std::vector<uint32_t>& vPatchSize, PatchType ePatchType, const std::vector<uint32_t>& vSlidingDis, bool bFreeLoadedData);
	virtual void CreateMask(MaskRandMode eRandMode, float fNonZeroRatio);
	virtual void Resize(size_t iNumPoints, const std::vector<size_t> vDims);
	virtual void ConvertTo1D(bool bDeleteOld);
	virtual void ConvertBack(bool bDeleteOld);
	virtual void Clamp(T minVal, T maxVal);
	virtual void RemoveZeros(CCS_INTERNAL_TYPE thresh);
	virtual bool AssembleData();
	virtual void CalcQuality(const CData<T, 4>* pOther, ReconQualityMetric eMetric, std::vector<CCS_INTERNAL_TYPE>& vQlty);
	virtual bool WriteAssembled(const std::string& strOutputFolder);
	virtual void CleanUp();

protected:

	uint32_t m_iPhi;	//Number of angular samples for Phi (azimuth)
	uint32_t m_iTheta;	//Number of angular samples for Theta (elevation)

	std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> m_vSxV;		// npc = 101

	std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> m_vUr;		// npc = 101
	std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> m_vUg;		// npc = 101
	std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> m_vUb;		// npc = 101	


private:

	//File names for BTFs
	std::vector<boost::filesystem::path> m_vFileNames;

	//Number of patches per dataset
	std::vector<size_t> m_vNumPatches;

	//Image size
	std::vector<Eigen::Vector2i> m_vBTFSize;

	//A vector storing sliding patch indices for each image (used for assembling sliding patches)
	std::vector<std::vector<Eigen::Vector2i> > m_vSlidingPatches;

	std::vector<std::vector<float> > m_vMean;
	std::vector<std::vector<float> > m_vStd;

	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_vBTFC1;              // U1 x SxV1'
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_vBTFC2;              // U2 x SxV2'
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_vBTFC3;              // U3 x SxV3'

	//Sampling Percentage for the training
	int m_iChannel;
};



#include "DataBTF4D.inl"


