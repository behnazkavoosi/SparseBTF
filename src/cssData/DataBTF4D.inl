#include "DataBTF4D.h"



namespace fs = ::boost::filesystem;



template <typename T>
CDataBTF4D<T>::CDataBTF4D(uint32_t iPhi, uint32_t iTheta) : CData<T, 4>()
{
	this->m_eDataType = CData<T, 4>::DataType::CCS_DATA_nD;
	m_iPhi = iPhi;
	m_iTheta = iTheta;
}


template <typename T>
CDataBTF4D<T>::~CDataBTF4D()
{
	CleanUp();
}


template<typename T>
size_t CDataBTF4D<T>::GetNumDataElems() const
{
	if (this->m_eDataType == CData<T, 4>::DataType::CCS_DATA_1D)
		return this->m_mData1D->size();
	else
		return this->m_mDatanD->size();
}

template <typename T>
void CDataBTF4D<T>::GetDataElemDim(std::vector<size_t>& vDims) const
{
	if (this->m_eDataType == CData<T, 4>::DataType::CCS_DATA_1D)
	{
		vDims.clear();
		vDims.resize(1);
		vDims[0] = this->m_vPatchSize[0] * this->m_vPatchSize[1] * m_iPhi * m_iTheta;
	}
	else
	{
		vDims.clear();
		vDims.resize(4);
		vDims[0] = this->m_vPatchSize[0];
		vDims[1] = this->m_vPatchSize[1];
		vDims[2] = m_iPhi;
		vDims[3] = m_iTheta;
	}
}

template<typename T>
inline void CDataBTF4D<T>::SetColorCh(int iChannel)
{
	this->m_iChannel = iChannel;
}


template <typename T>
bool CDataBTF4D<T>::Init()
{
	CleanUp();

	return true;
}


template <typename T>
bool CDataBTF4D<T>::LoadFromDisk(const std::string& strDataFolder)
{
	this->m_strDataFolder = strDataFolder;

	fs::path rootPath(this->m_strDataFolder);
	if (!fs::exists(rootPath) || !fs::is_directory(rootPath))
	{
		std::cerr << "ERROR: The input folder does not exist." << std::endl;
		return false;
	}

	//Find BTFs in the root folder
	m_vFileNames.clear();
	fs::directory_iterator it_end;
	for (fs::directory_iterator it(rootPath); it != it_end; ++it)
		if (fs::is_regular_file(it->status()))
			m_vFileNames.push_back(it->path().filename());

	//Sort BTF file names
	std::sort(m_vFileNames.begin(), m_vFileNames.end(), [](const fs::path& p1, const fs::path& p2) {return p1.filename().string() < p2.filename().string(); });

	size_t iImSize = 400;  //400
	size_t iAngSize = 151; //151
	std::vector<size_t> vBTFdims = { iImSize * iImSize, iAngSize * iAngSize };

	size_t iNpcy = 101; //101

	m_vSxV.resize(m_vFileNames.size(), Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(vBTFdims[0], iNpcy));

	m_vUr.resize(m_vFileNames.size(), Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(vBTFdims[1], iNpcy));
	m_vUg.resize(m_vFileNames.size(), Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(vBTFdims[1], iNpcy));
	m_vUb.resize(m_vFileNames.size(), Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(vBTFdims[1], iNpcy));


	for (size_t i = 0; i < m_vFileNames.size(); ++i)
	{
		VclMatio matReader;

		if (!matReader.openForReading((rootPath / m_vFileNames[i]).string()))
		{
			std::cerr << "ERROR: Cannot read BTF file at " << (rootPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		if (!matReader.readEigenMatrix2dNamed("SxV", m_vSxV[i]))
		{
			std::cerr << "ERROR: Cannot read SxV variable at " << (rootPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		if (!matReader.readEigenMatrix2dNamed("U1", m_vUr[i]))
		{
			std::cerr << "ERROR: Cannot read U variable at " << (rootPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		if (!matReader.readEigenMatrix2dNamed("U2", m_vUg[i]))
		{
			std::cerr << "ERROR: Cannot read U variable at " << (rootPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		if (!matReader.readEigenMatrix2dNamed("U3", m_vUb[i]))
		{
			std::cerr << "ERROR: Cannot read U variable at " << (rootPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		matReader.close();
		std::cout << "loaded" << std::endl;
	}

	m_vBTFSize.resize(m_vFileNames.size());
	for (size_t i = 0; i < m_vFileNames.size(); ++i)
		m_vBTFSize[i] = Eigen::Vector2i(iImSize, iImSize);

	return true;
}


template <typename T>
bool CDataBTF4D<T>::PrepareData(const std::vector<uint32_t>& vPatchSize, PatchType ePatchType, const std::vector<uint32_t>& vSlidingDis, bool bFreeLoadedData)
{
	this->m_vPatchSize = vPatchSize;
	this->m_ePatchType = ePatchType;
	this->m_vSlidingDis = vSlidingDis;

	//Validate parameters
	if (vPatchSize.size() != 2)
	{
		std::cerr << "ERROR: Invalid patch size" << std::endl;
		return false;
	}

	if ((m_vSxV.size() == 0) || (m_vUr.size() == 0) || (m_vUg.size() == 0) || (m_vUb.size() == 0) || (m_vBTFSize.size() == 0))
	{
		std::cerr << "ERROR: Data not loaded yet. Cannot prepare data." << std::endl;
		return false;
	}

	Eigen::Vector2i vPS(this->m_vPatchSize[0], this->m_vPatchSize[1]);
	size_t iStart = 0;
	size_t iSpatialPatchSize = vPS.prod();
	size_t iNumAng = m_iTheta * m_iPhi;

	//Calculate total number of patches
	std::vector<size_t> vDims;
	GetDataElemDim(vDims);
	m_vNumPatches.resize(m_vBTFSize.size());
	m_vSlidingPatches.resize(m_vBTFSize.size());
	for (size_t i = 0; i < m_vNumPatches.size(); ++i)
	{
		//m_vNumPatches[i] = m_vBTFSize[i].prod() / (iSpatialPatchSize * 16); // cropped BTF: 100x100

		m_vNumPatches[i] = m_vBTFSize[i].prod() / iSpatialPatchSize; // full BTF: 400x400
	}
	size_t iTotalNumPatches = std::accumulate(m_vNumPatches.begin(), m_vNumPatches.end(), 0);

	//Allocate space
	SAFE_DELETE(this->m_mDatanD);
	this->m_mDatanD = new std::vector<typename CData<T, 4>::ArraynD>(iTotalNumPatches, typename CData<T, 4>::ArraynD(vDims, CCS_TENSOR_STORAGE_ORDER));

	m_vMean.resize(m_vBTFSize.size());
	m_vStd.resize(m_vBTFSize.size());
	
	for (size_t i = 0; i < m_vBTFSize.size(); ++i)
	{
		Eigen::Vector2i vImagePS(m_vBTFSize[i][0], m_vBTFSize[i][1]);
		if (i > 0)
			iStart += m_vNumPatches[i - 1];

		m_vBTFC1.resize(m_vBTFSize[i].prod(), iNumAng);
		m_vBTFC2.resize(m_vBTFSize[i].prod(), iNumAng);
		m_vBTFC3.resize(m_vBTFSize[i].prod(), iNumAng);

		m_vBTFC1 = m_vSxV[i] * m_vUr[i].transpose();
		m_vBTFC2 = m_vSxV[i] * m_vUg[i].transpose();
		m_vBTFC3 = m_vSxV[i] * m_vUb[i].transpose();

		m_vMean[i].resize(iNumAng);
		m_vStd[i].resize(iNumAng);

#pragma omp parallel for schedule(dynamic,1)
		for (int j = 0; j < iNumAng; ++j)
		{
			std::vector<Eigen::Vector2i> tmpSlidingPatches;
			size_t mblocks, nblocks;

			//Convert columns to images (done for each color channel)
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mPatchesY, mPatchesU, mPatchesV;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mImageY, mImageU, mImageV;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mCropImageY, mCropImageU, mCropImageV;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mTmpY, mTmpU, mTmpV;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mPermY, mPermU, mPermV;


			mImageY.resize(m_vBTFSize[i][0], m_vBTFSize[i][1]);
			mImageU.resize(m_vBTFSize[i][0], m_vBTFSize[i][1]);
			mImageV.resize(m_vBTFSize[i][0], m_vBTFSize[i][1]);

			mPermY.resize(m_vBTFSize[i][1], m_vBTFSize[i][0]);
            mPermU.resize(m_vBTFSize[i][1], m_vBTFSize[i][0]);
            mPermV.resize(m_vBTFSize[i][1], m_vBTFSize[i][0]);

			mTmpY.resize(m_vBTFSize[i][0], m_vBTFSize[i][1]);
			mTmpU.resize(m_vBTFSize[i][0], m_vBTFSize[i][1]);
			mTmpV.resize(m_vBTFSize[i][0], m_vBTFSize[i][1]);

			if (!col2im<T>(m_vBTFC1.col(j), mImageY, vImagePS, m_vBTFSize[i], this->m_ePatchType, m_vSlidingPatches[i]))
				std::cerr << "ERROR: col2im() error." << std::endl;
			if (!col2im<T>(m_vBTFC2.col(j), mImageU, vImagePS, m_vBTFSize[i], this->m_ePatchType, m_vSlidingPatches[i]))
				std::cerr << "ERROR: col2im() error." << std::endl;
			if (!col2im<T>(m_vBTFC3.col(j), mImageV, vImagePS, m_vBTFSize[i], this->m_ePatchType, m_vSlidingPatches[i]))
				std::cerr << "ERROR: col2im() error." << std::endl;

			/*for (int l = 0; l < m_vBTFSize[i][0]; ++l)
            {
                                for (int z = 0; z < m_vBTFSize[i][1]; ++z)
                                {
                                        mPermY(l, z) = mImageY(z, l);
                                        mPermU(l, z) = mImageU(z, l);
                                        mPermV(l, z) = mImageV(z, l);
                                }
            }*/			


			mTmpY = 0.299 * mImageY.array() + 0.587 * mImageU.array() + 0.114 * mImageV.array();
			mTmpU = -0.14713 * mImageY.array() - 0.28886 * mImageU.array() + 0.436 * mImageV.array();
			mTmpV = 0.615 * mImageY.array() - 0.51499 * mImageU.array() - 0.10001 * mImageV.array();


			float fMean1, fMean2, fMean3 = 0;
			fMean1 = mTmpY.array().mean();
			fMean2 = mTmpU.array().mean();
			fMean3 = mTmpV.array().mean();

			m_vMean[i][j] = (fMean1 + fMean2 + fMean3) / 3;

			float fStd1, fStd2, fStd3 = 0;
			fStd1 = sqrt((mTmpY.array() - m_vMean[i][j]).square().sum() / (mTmpY.size() - 1));
			fStd2 = sqrt((mTmpU.array() - m_vMean[i][j]).square().sum() / (mTmpU.size() - 1));
			fStd3 = sqrt((mTmpV.array() - m_vMean[i][j]).square().sum() / (mTmpV.size() - 1));

			m_vStd[i][j] = (fStd1 + fStd2 + fStd3) / 3;

			mTmpY = ((mTmpY.array() - m_vMean[i][j]) / m_vStd[i][j]).eval();
			mTmpU = ((mTmpU.array() - m_vMean[i][j]) / m_vStd[i][j]).eval();
			mTmpV = ((mTmpV.array() - m_vMean[i][j]) / m_vStd[i][j]).eval();

			if (!im2col<T>(mTmpY, vPS, this->m_ePatchType, this->m_vSlidingDis, tmpSlidingPatches, 0.0, mPatchesY, mblocks, nblocks))
				std::cerr << "ERROR: im2col() error." << std::endl;
			if (!im2col<T>(mTmpU, vPS, this->m_ePatchType, this->m_vSlidingDis, tmpSlidingPatches, 0.0, mPatchesU, mblocks, nblocks))
				std::cerr << "ERROR: im2col() error." << std::endl;
			if (!im2col<T>(mTmpV, vPS, this->m_ePatchType, this->m_vSlidingDis, tmpSlidingPatches, 0.0, mPatchesV, mblocks, nblocks))
				std::cerr << "ERROR: im2col() error." << std::endl;

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mPatch(iSpatialPatchSize, m_vNumPatches[i]);

			for (size_t k = 0; k < iSpatialPatchSize; ++k)
			{
				if (m_iChannel == 0)
					mPatch.row(k) = mPatchesY.row(k);
				else if (m_iChannel == 1)
					mPatch.row(k) = mPatchesU.row(k);
				else if (m_iChannel == 2)
					mPatch.row(k) = mPatchesV.row(k);

			}

			for (size_t p = 0; p < m_vNumPatches[i]; ++p)
			{
				for (size_t m = 0; m < vDims[1]; ++m)
				{
					for (size_t n = 0; n < vDims[0]; ++n)
					{
						(this->m_mDatanD->at(iStart + p))[n][m][j % m_iPhi][j / m_iPhi] = mPatch(n + m * vDims[0], p);
					}
				}

			}
		}
		std::cout << "prepared" << std::endl;
	}

	return true;
}

template<typename T>
inline void CDataBTF4D<T>::CreateMask(MaskRandMode eRandMode, float fNonZeroRatio)
{
}


template <typename T>
void CDataBTF4D<T>::Resize(size_t iNumPoints, const std::vector<size_t> vDims)
{
	if (this->m_eDataType == CData<T, 4>::DataType::CCS_DATA_1D)
	{
		size_t iLength = std::accumulate(vDims.begin(), vDims.end(), 1, std::multiplies<size_t>());
		SAFE_DELETE(this->m_mData1D);
		this->m_mData1D = new std::vector<typename CData<T, 4>::Array1D>(iNumPoints);
		for (size_t i = 0; i < this->m_mData1D->size(); ++i)
			this->m_mData1D->at(i).setZero(iLength);
	}
	else
	{
		SAFE_DELETE(this->m_mDatanD);
		this->m_mDatanD = new std::vector<typename CData<T, 4>::ArraynD>(iNumPoints, typename CData<T, 4>::ArraynD(vDims, CCS_TENSOR_STORAGE_ORDER));
		for (size_t i = 0; i < this->m_mDatanD->size(); ++i)
			TensorSetZero(this->m_mDatanD->at(i));
	}
}


template <typename T>
template <typename U>
void CDataBTF4D<T>::ResizeLike(const CData<U, 4>* pData)
{
	CleanUp();
	CDataBTF4D<U>* pDataBTF = dynamic_cast<CDataBTF4D<U>*>(const_cast<CData<U, 4>*>(pData));
	if (!pDataBTF)
		return;

	CopyPropsFrom(pDataBTF);
	std::vector<size_t> vDims;
	pDataBTF->GetDataElemDim(vDims);
	if (pDataBTF->GetDataType() == CData<T, 4>::DataType::CCS_DATA_1D)
	{
		size_t iLength = std::accumulate(vDims.begin(), vDims.end(), 1, std::multiplies<size_t>());
		SAFE_DELETE(this->m_mData1D);
		this->m_mData1D = new std::vector<typename CData<T, 4>::Array1D>(pDataBTF->GetNumDataElems());
		for (size_t i = 0; i < this->m_mData1D->size(); ++i)
			this->m_mData1D->at(i).setZero(iLength);
	}
	else
	{
		SAFE_DELETE(this->m_mDatanD);
		this->m_mDatanD = new std::vector<typename CData<T, 4>::ArraynD>(pDataBTF->GetNumDataElems(), typename CData<T, 4>::ArraynD(vDims, CCS_TENSOR_STORAGE_ORDER));
		for (size_t i = 0; i < this->m_mDatanD->size(); ++i)
			TensorSetZero(this->m_mDatanD->at(i));
	}
}


template <typename T>
template <typename U>
void CDataBTF4D<T>::CopyPropsFrom(const CData<U, 4>* pData)
{
	const CDataBTF4D<U>* pDataBTF = dynamic_cast<const CDataBTF4D<U>*>(pData);
	if (!pDataBTF)
		return;
	m_vFileNames = pDataBTF->m_vFileNames;
	m_vNumPatches = pDataBTF->m_vNumPatches;
	m_vBTFSize = pDataBTF->m_vBTFSize;
	m_vSlidingPatches = pDataBTF->m_vSlidingPatches;
	m_iChannel = pDataBTF->m_iChannel;
	m_vStd = pDataBTF->m_vStd;
	m_vMean = pDataBTF->m_vMean;
	this->m_eDataType = pDataBTF->m_eDataType;
	this->m_vPatchSize = pDataBTF->m_vPatchSize;
	this->m_ePatchType = pDataBTF->m_ePatchType;
	this->m_vSlidingDis = pDataBTF->m_vSlidingDis;
	this->m_iQuantization = pDataBTF->m_iQuantization;
	this->m_strDataFolder = pDataBTF->m_strDataFolder;
	this->m_strOutputFolder = pDataBTF->m_strOutputFolder;
	this->m_vMean = pDataBTF->m_vMean;
	this->m_vStd = pDataBTF->m_vStd;

}


template <typename T>
void CDataBTF4D<T>::ConvertTo1D(bool bDeleteOld)
{
	size_t iNumPoints = GetNumDataElems();
	std::vector<size_t> vDims;
	GetDataElemDim(vDims);
	size_t iLength = std::accumulate(vDims.begin(), vDims.end(), 1, std::multiplies<size_t>());
	this->m_eDataType = CData<T, 4>::DataType::CCS_DATA_1D;
	SAFE_DELETE(this->m_mData1D);
	this->m_mData1D = new std::vector<typename CData<T, 4>::Array1D>(iNumPoints);
	for (size_t i = 0; i < iNumPoints; ++i)
	{
		this->m_mData1D->at(i).resize(iLength);
		for (size_t j = 0; j < iLength; ++j)
			(this->m_mData1D->at(i))(j) = (this->m_mDatanD->at(i).data())[j];
	}
	if (bDeleteOld)
		SAFE_DELETE(this->m_mDatanD);

	if (this->m_mMasknD)
	{
		SAFE_DELETE(this->m_mMask1D);
		this->m_mMask1D = new std::vector<typename CData<T, 4>::Msk1D>(iNumPoints);
		for (size_t i = 0; i < iNumPoints; ++i)
		{
			this->m_mMask1D->at(i).resize(iLength);
			for (size_t j = 0; j < iLength; ++j)
				(this->m_mMask1D->at(i))(j) = (this->m_mMasknD->at(i).data())[j];
		}
		if (bDeleteOld)
			SAFE_DELETE(this->m_mMasknD);
	}
}


template<typename T>
void CDataBTF4D<T>::ConvertBack(bool bDeleteOld)
{
	size_t iNumPoints = GetNumDataElems();
	this->m_eDataType = CData<T, 4>::DataType::CCS_DATA_nD;
	std::vector<size_t> vDims;
	GetDataElemDim(vDims);
	size_t iLength = std::accumulate(vDims.begin(), vDims.end(), 1, std::multiplies<size_t>());
	SAFE_DELETE(this->m_mDatanD);
	this->m_mDatanD = new std::vector<typename CData<T, 4>::ArraynD>(iNumPoints, typename CData<T, 4>::ArraynD(vDims, CCS_TENSOR_STORAGE_ORDER));
	for (size_t i = 0; i < iNumPoints; ++i)
		for (size_t j = 0; j < iLength; ++j)
			(this->m_mDatanD->at(i).data())[j] = (this->m_mData1D->at(i))(j);
	if (bDeleteOld)
		SAFE_DELETE(this->m_mData1D);

	if (this->m_mMask1D)
	{
		SAFE_DELETE(this->m_mMasknD);
		this->m_mMasknD = new std::vector<typename CData<T, 4>::MsknD>(iNumPoints, typename CData<T, 4>::MsknD(vDims, CCS_TENSOR_STORAGE_ORDER));
		for (size_t i = 0; i < iNumPoints; ++i)
			for (size_t j = 0; j < iLength; ++j)
				(this->m_mMasknD->at(i).data())[j] = (this->m_mMask1D->at(i))(j);
		if (bDeleteOld)
			SAFE_DELETE(this->m_mMask1D);
	}
}


template <typename T>
void CDataBTF4D<T>::Clamp(T minVal, T maxVal)
{
	if (this->m_eDataType == CData<T, 4>::DataType::CCS_DATA_1D)
	{
#pragma omp parallel for schedule(dynamic,1)
		for (int i = 0; i < GetNumDataElems(); ++i)
			Clamp1D(this->m_mData1D->at(i), minVal, maxVal);
	}
	else
	{
#pragma omp parallel for schedule(dynamic,1)
		for (int i = 0; i < GetNumDataElems(); ++i)
			ClampnD(this->m_mDatanD->at(i), minVal, maxVal);
	}
}


template <typename T>
void CDataBTF4D<T>::RemoveZeros(CCS_INTERNAL_TYPE thresh)
{
	if (thresh < 0.0)
		return;

	std::vector<size_t> vDims;
	GetDataElemDim(vDims);
	size_t iLength = std::accumulate(vDims.begin(), vDims.end(), 1, std::multiplies<size_t>());
	std::vector<size_t> vIdx;
	vIdx.reserve(GetNumDataElems());
	for (size_t i = 0; i < GetNumDataElems(); ++i)
	{
		typename CData<T, 4>::InternalArraynD dataPoint(vDims, CCS_TENSOR_STORAGE_ORDER);
		CastDataPointTTypeToInternal<T, 4>(this->m_mDatanD->at(i), dataPoint);
		if (TensorNorm2(dataPoint) > thresh)
			vIdx.push_back(i);
	}

	std::vector<typename CData<T, 4>::ArraynD>* mDatanD = new std::vector<typename CData<T, 4>::ArraynD>(vIdx.size(), typename CData<T, 4>::ArraynD(vDims, CCS_TENSOR_STORAGE_ORDER));
	for (size_t i = 0; i < vIdx.size(); ++i)
		for (size_t j = 0; j < iLength; ++j)
			(mDatanD->at(i).data())[j] = (this->m_mDatanD->at(vIdx[i]).data())[j];
	SAFE_DELETE(this->m_mDatanD);
	this->m_mDatanD = mDatanD;
}


template <typename T>
bool CDataBTF4D<T>::AssembleData()
{
	std::string strOutputFolder;
	
	if (m_iChannel == 0)
		strOutputFolder = "./Y/";
	else if (m_iChannel == 1)
		strOutputFolder = "./U/";
	else if (m_iChannel == 2)
		strOutputFolder = "./V/";


	if (m_vFileNames.empty() || m_vNumPatches.empty() || m_vBTFSize.empty())
		return false;

	Eigen::Vector2i vPS(this->m_vPatchSize[0], this->m_vPatchSize[1]);
	size_t iStart = 0;
	size_t iSpatialPatchSize = vPS.prod();
	size_t iNumAng = m_iTheta * m_iPhi;

	//Calculate total number of patches
	std::vector<size_t> vDims;
	GetDataElemDim(vDims);

	for (size_t i = 0; i < m_vBTFSize.size(); ++i)
	{
		Eigen::Vector2i vImagePS(m_vBTFSize[i][0], m_vBTFSize[i][1]);
		Eigen::Vector2i vImSize(m_vBTFSize[i][0], m_vBTFSize[i][1]);

		if (i > 0)
			iStart += m_vNumPatches[i - 1];

		m_vBTFC1.resize(m_vBTFSize[i].prod(), iNumAng); //400000, 22801

#pragma omp parallel for schedule(dynamic,1)
		for (int j = 0; j < iNumAng; ++j)
		{
			std::vector<Eigen::Vector2i> tmpSlidingPatches;
			size_t mblocks, nblocks;

			//Convert columns to images (done for each color channel)
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mPatchesY, mTmpY, mColY, mImageY, mPermY;

			mPermY.resize(m_vBTFSize[i][1], m_vBTFSize[i][0]);
			mImageY.resize(m_vBTFSize[i][0], m_vBTFSize[i][1]);
			mTmpY.resize(m_vBTFSize[i][0], m_vBTFSize[i][1]);
			mPatchesY.resize(iSpatialPatchSize, m_vNumPatches[i]);

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mPatch(iSpatialPatchSize, m_vNumPatches[i]);

			for (size_t p = 0; p < m_vNumPatches[i]; ++p)
			{
				for (size_t m = 0; m < vDims[1]; ++m)
				{
					for (size_t n = 0; n < vDims[0]; ++n)
					{
						mPatch(n + m * vDims[0], p) = (this->m_mDatanD->at(iStart + p))[n][m][j % m_iPhi][j / m_iPhi];
					}
				}

			}

			for (size_t k = 0; k < iSpatialPatchSize; ++k)
				mPatchesY.row(k) = mPatch.row(k);


			//Convert patches to images (done for each color channel)
			if (!col2im<T>(mPatchesY, mImageY, vPS, vImSize, this->m_ePatchType, m_vSlidingPatches[i]))
				std::cerr << "ERROR: col2im() error." << std::endl;

			mImageY = (mImageY.array() * m_vStd[i][j] + m_vMean[i][j]).eval();
			 
			if (!im2col<T>(mImageY, vImagePS, this->m_ePatchType, this->m_vSlidingDis, tmpSlidingPatches, 0.0, mColY, mblocks, nblocks))
				std::cerr << "ERROR: im2col() error." << std::endl;

			m_vBTFC1.col(j) = mColY;

		}

		this->m_strOutputFolder = strOutputFolder;
		fs::path outputPath(this->m_strOutputFolder);
		if (!fs::exists(outputPath) || !fs::is_directory(outputPath))
		{
			if (!fs::create_directory(outputPath))
			{
				std::cerr << "ERROR: Unable to create output directory to store results." << std::endl;
				return false;
			}
		}

		VclMatio outFile;

		if (!outFile.openForWriting((outputPath / m_vFileNames[i]).string()))
		{
			std::cerr << "ERROR: Cannot write BTF file at " << (outputPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		if (!outFile.writeEigenMatrix2dNamed("btf1", m_vBTFC1))
		{
			std::cerr << "ERROR: Cannot write BTF variable at " << (outputPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		outFile.close();
	}

	return true;
}



template<typename T>
void CDataBTF4D<T>::CalcQuality(const CData<T, 4>* pOther, ReconQualityMetric eMetric, std::vector<CCS_INTERNAL_TYPE>& vQlty)
{
}



template <typename T>
bool CDataBTF4D<T>::WriteAssembled(const std::string& strOutputFolder)
{
	if (m_vFileNames.empty() || m_vNumPatches.empty() || m_vBTFSize.empty())
		return false;

	//Create the output folder (if there is none)
	this->m_strOutputFolder = strOutputFolder;
	fs::path outputPath(this->m_strOutputFolder);
	if (!fs::exists(outputPath) || !fs::is_directory(outputPath))
	{
		if (!fs::create_directory(outputPath))
		{
			std::cerr << "ERROR: Unable to create output directory to store results." << std::endl;
			return false;
		}
	}

	//Create output folders for each data set
	for (size_t i = 0; i < m_vFileNames.size(); ++i)
	{
		VclMatio outFile;

		if (!outFile.openForWriting((outputPath / m_vFileNames[i]).string()))
		{
			std::cerr << "ERROR: Cannot write BTF file at " << (outputPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		if (!outFile.writeEigenMatrix2dNamed("btf1", m_vBTFC1))
		{
			std::cerr << "ERROR: Cannot write BTF variable at " << (outputPath / m_vFileNames[i]).string() << std::endl;
			return false;
		}
		outFile.close();
	}

	return true;
}


template <typename T>
void CDataBTF4D<T>::CleanUp()
{
	SAFE_DELETE(this->m_mData1D);
	SAFE_DELETE(this->m_mData2D);
	SAFE_DELETE(this->m_mDatanD);
	SAFE_DELETE(this->m_mMask1D);
	SAFE_DELETE(this->m_mMask2D);
	SAFE_DELETE(this->m_mMasknD);
	this->m_strDataFolder.clear();
	this->m_strOutputFolder.clear();
	this->m_ePatchType = CCS_PATCHTYPE_NOV;
	this->m_vSlidingDis.clear();
	this->m_iQuantization = 0;
	this->m_vPatchSize.clear();

	m_vSxV.clear();
	m_vUr.clear();
	m_vUg.clear();
	m_vUb.clear();
	m_vBTFC1.resize(0, 0);              // U1 x SxV1'
	m_vBTFC2.resize(0, 0);              // U2 x SxV2'
	m_vBTFC3.resize(0, 0);              // U3 x SxV3'

	m_iChannel = 0;
	m_vFileNames.clear();
	m_vNumPatches.clear();
	m_vBTFSize.clear();
	m_vSlidingPatches.clear();
}







