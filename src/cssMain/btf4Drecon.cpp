


#include "cssData/DataBTF4D.h"
#include "cssDict/DictTestEnsOrthnD.h"
#include "cssRecon/ReconEnsOrthnD.h"



#if defined(_MSC_VER) && _MSC_VER >= 1400 
#pragma warning(push) 
#pragma warning(disable:4996) 
#endif 



int main(int argc, char* argv[])
{
	omp_set_dynamic(0);
	omp_set_nested(1);
	//Initialize MPI for multi-computer parallelization 
	//Each node will be parallelized using OpenMP
	int iThreadLevelRequired = MPI_THREAD_SERIALIZED;
	int iThreadLevelProvided = 0;
	int iNumNodes = 0;
	int iNodeID = 0;
	MPI_Init_thread(NULL, NULL, iThreadLevelRequired, &iThreadLevelProvided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumNodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &iNodeID);
	if (iThreadLevelProvided < iThreadLevelRequired)
	{
		if (iNodeID == 0)
			std::cout << "Warning: This MPI implementation provides insufficient threading support." << std::endl;
		return -1;
	}
	int iLen = 0;
	char strHostname[MPI_MAX_PROCESSOR_NAME];
	if (iNodeID == 0)
		std::cout << "Number of nodes is " << iNumNodes << std::endl;
	MPI_Get_processor_name(strHostname, &iLen);
	std::cout << "Node ID " << iNodeID << " on " << "<" << strHostname << ">" << " with " << omp_get_max_threads() << " threads reporting for duty!" << std::endl;
	double dTimeDelta = 0.0;

	typedef float DataType;

	std::vector<uint32_t> vPatchSize(2);
	vPatchSize[0] = 10;
	vPatchSize[1] = 10;

	std::vector<uint32_t> vSlidingDist(2);
	vSlidingDist[0] = 1;
	vSlidingDist[1] = 1;
	
	float fSampling = 1;
	uint32_t iSparsity = 885;  
	CCS_INTERNAL_TYPE threshold = 1e-5; 

	std::string strTestSetDir = "./dataset";
	std::string strCoeffsDir = "./Coeff_files/";
	std::string strDataSetName = "set1";
	std::string strDictAddrR = "./DictEnsOrth4DY.mat";
	std::string strCoeffsFilenameR = "Coeffs_BTF_set1_885_1e-05_Y.mat";

	std::string strDictAddrG = "./DictEnsOrth4DU.mat";
	std::string strCoeffsFilenameG = "Coeffs_BTF_set1_64_1e-05_U.mat";

	std::string strDictAddrB = "./DictEnsOrth4DV.mat";
	std::string strCoeffsFilenameB = "Coeffs_BTF_set1_64_1e-05_V.mat";

	CDictTestEnsOrthnD<DataType, 4>* pTest = NULL;
	CDictionary* pDict = NULL;
	CDataBTF4D<DataType>* pTestData = NULL;
	CReconEnsOrthnD<DataType, 4>* pRecon = NULL;
	CDataBTF4D<DataType>* pOutputData = NULL;
	
	pTestData = new CDataBTF4D<DataType>(151, 151);
	if (!pTestData->Init()) return -1;
	if (!pTestData->LoadFromDisk(strTestSetDir)) return -1;
	pTestData->SetColorCh(0);
	if (!pTestData->PrepareData(vPatchSize, CCS_PATCHTYPE_NOV, vSlidingDist, true)) return -1;
	std::cout << "Number of patches: " << pTestData->GetNumDataElems() << std::endl;

	pDict = new CDictionary();
	if (!pDict->Load(strDictAddrR)) return -1;
	pTest = new CDictTestEnsOrthnD<DataType, 4>();
	if (!pTest->Init(true)) return -1;
	if (!pTest->Load(strCoeffsDir, strCoeffsFilenameR, true))       return -1;
	pRecon = new CReconEnsOrthnD<DataType, 4>();
	pOutputData = new CDataBTF4D<DataType>(151, 151);
	pOutputData->ResizeLike(pTestData);
	if (!pRecon->Init())    return -1;
	std::cout << "Reconstructing" << std::endl;
	if (!pRecon->Reconstruct(pTest, pDict, iSparsity, threshold, pOutputData, dTimeDelta))  return -1;
	std::cout << "Done in " << dTimeDelta << " seconds" << std::endl;
	if (!pOutputData->AssembleData()) return -1;

	std::cout << "MSE: " << MSE<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;
	std::cout << "PSNR: " << PSNR<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;
	std::cout << "SNR: " << SNR<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;

	SAFE_DELETE(pOutputData);
	SAFE_DELETE(pRecon);
	SAFE_DELETE(pDict);
	SAFE_DELETE(pTest);
	SAFE_DELETE(pTestData);

	iSparsity = 128;

	pTestData = new CDataBTF4D<DataType>(151, 151);
	if (!pTestData->Init()) return -1;
	if (!pTestData->LoadFromDisk(strTestSetDir)) return -1;
	pTestData->SetColorCh(1);
	if (!pTestData->PrepareData(vPatchSize, CCS_PATCHTYPE_NOV, vSlidingDist, true)) return -1;
	std::cout << "Number of patches: " << pTestData->GetNumDataElems() << std::endl;

	pDict = new CDictionary();
	if (!pDict->Load(strDictAddrG)) return -1;
	pTest = new CDictTestEnsOrthnD<DataType, 4>();
	if (!pTest->Init(true)) return -1;
	if (!pTest->Load(strCoeffsDir, strCoeffsFilenameG, true))       return -1;
	pRecon = new CReconEnsOrthnD<DataType, 4>();
	pOutputData = new CDataBTF4D<DataType>(151, 151);
	pOutputData->ResizeLike(pTestData);
	if (!pRecon->Init())    return -1;
	std::cout << "Reconstructing" << std::endl;
	if (!pRecon->Reconstruct(pTest, pDict, iSparsity, threshold, pOutputData, dTimeDelta))  return -1;
	std::cout << "Done in " << dTimeDelta << " seconds" << std::endl;
	if (!pOutputData->AssembleData()) return -1;

	std::cout << "MSE: " << MSE<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;
	std::cout << "PSNR: " << PSNR<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;
	std::cout << "SNR: " << SNR<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;

	SAFE_DELETE(pOutputData);
	SAFE_DELETE(pRecon);
	SAFE_DELETE(pDict);
	SAFE_DELETE(pTest);
	SAFE_DELETE(pTestData);

	pTestData = new CDataBTF4D<DataType>(151, 151);
	if (!pTestData->Init()) return -1;
	if (!pTestData->LoadFromDisk(strTestSetDir)) return -1;
	pTestData->SetColorCh(2);
	if (!pTestData->PrepareData(vPatchSize, CCS_PATCHTYPE_NOV, vSlidingDist, true)) return -1;
	std::cout << "Number of patches: " << pTestData->GetNumDataElems() << std::endl;

	pDict = new CDictionary();
	if (!pDict->Load(strDictAddrB)) return -1;
	pTest = new CDictTestEnsOrthnD<DataType, 4>();
	if (!pTest->Init(true)) return -1;
	if (!pTest->Load(strCoeffsDir, strCoeffsFilenameB, true))       return -1;
	pRecon = new CReconEnsOrthnD<DataType, 4>();
	pOutputData = new CDataBTF4D<DataType>(151, 151);
	pOutputData->ResizeLike(pTestData);
	if (!pRecon->Init())    return -1;
	std::cout << "Reconstructing" << std::endl;
	if (!pRecon->Reconstruct(pTest, pDict, iSparsity, threshold, pOutputData, dTimeDelta))  return -1;
	std::cout << "Done in " << dTimeDelta << " seconds" << std::endl;
	if (!pOutputData->AssembleData()) return -1;

	std::cout << "MSE: " << MSE<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;
	std::cout << "PSNR: " << PSNR<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;
	std::cout << "SNR: " << SNR<DataType>(*pTestData->GetDatanD(), *pOutputData->GetDatanD()) << std::endl;

	SAFE_DELETE(pOutputData);
	SAFE_DELETE(pRecon);
	SAFE_DELETE(pDict);
	SAFE_DELETE(pTest);
	SAFE_DELETE(pTestData);


	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	int x;
	std::cin >> x;

	return 0;
}
