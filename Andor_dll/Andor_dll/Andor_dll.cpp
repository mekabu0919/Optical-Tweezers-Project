// Andor_dll.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//
//#ifdef _DEBUG
//#define CV_EXT "d.lib"
//#else
//#define CV_EXT ".lib"
//#endif
//#pragma comment(lib, "opencv_world340" CV_EXT) // OpenCV3.3.0の場合は、"opencv_core330"に
//#pragma comment(lib, "opencv_highgui320" CV_EXT) // OpenCV3.3.0の場合は、"opencv_highgui330"に変更


#include "atcore.h"
#include "atutility.h"
#include "windows.h"
#include "RsDio.h"
#include <iostream>
#include <fstream>
#include <direct.h>
#include <iomanip>
#include <sstream>
#include <thread>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "E816_DLL.h"

using namespace std;

#define DLLEXPORT extern "C" __declspec(dllexport)
#define EXTRACTLOWPACKED(SourcePtr) ( (SourcePtr[0] << 4) + (SourcePtr[1] & 0xF) )
#define EXTRACTHIGHPACKED(SourcePtr) ( (SourcePtr[2] << 4) + (SourcePtr[1] >> 4) )

DLLEXPORT int initPiezo() {
	int ID = -1;
	char szDevices[1000];
	int nrDevices = E816_EnumerateUSB(szDevices, 999, NULL);
	if (nrDevices <= 0)
	{
		return -1;
	}
	else {
		ID = E816_ConnectUSB(szDevices);
		return ID;
	}
}

DLLEXPORT int finPiezo(int ID) {
	E816_CloseConnection(ID);
	return 0;
}

DLLEXPORT int setPiezoServo(int ID, BOOL checked) {
	BOOL vArray[1] = { checked };
	int ret = E816_SVO(ID, "A", vArray);
	return ret;
}

DLLEXPORT int movePiezo(int ID, double target) {
	double vArray[1] = { target };
	return E816_MOV(ID, "A", vArray);
}

DLLEXPORT int getPiezoPosition(int ID, double& pos) {
	double rArray[1];
	if (E816_qPOS(ID, "A", rArray)) {
		pos = rArray[0];
		return 0;
	}
	else {
		return -1;
	}
}

DLLEXPORT int testPiezo(int ID, double target, double &pos) {
	double vArray[1] = { target };
	double rArray[1];
	E816_qPOS(ID, "A", rArray);
	E816_MOV(ID, "A", vArray);
	pos = rArray[0];
	return 0;
}


DLLEXPORT int initDIO_shutter(HANDLE& handle) {
#ifdef WM_RSDIO
	handle = DioOpen(1);
	SetDirection(handle, 0);
	WritePort(handle, 0);
#endif // WM_RSDIO
	return 0;
}

DLLEXPORT int initDIO(HANDLE& handle) {
	#ifdef WM_RSDIO
		handle = DioOpen(0);
		SetDirection(handle, 0);
		WritePort(handle, 0);
	#endif // WM_RSDIO
	return 0;
}

DLLEXPORT int writeDIO(HANDLE handle, UCHAR val) {
#ifdef WM_RSDIO
	WritePort(handle, val);
#endif // WM_RSDIO
	return 0;
}

DLLEXPORT int finDIO(HANDLE handle) {
#ifdef WM_RSDIO
	DioClose(handle);
#endif // WM_RSDIO
	return 0;
}


DLLEXPORT int init(AT_H &Hndl) {
	int i_retCode;
//	i_retCode = AT_InitialiseLibrary();
//	if (i_retCode != AT_SUCCESS) {
//		//error condition, check atdebug.log file
//	}
	AT_64 iNumberDevices = 0;
	i_retCode = AT_GetInt(AT_HANDLE_SYSTEM, L"DeviceCount", &iNumberDevices);
	if (iNumberDevices <= 0) {
		// No cameras found, check all redistributable binaries
		// have been copied to the executable directory or are in the system path
		// and check atdebug.log file
	}
	else {
		i_retCode = AT_Open(0, &Hndl);
		if (i_retCode != AT_SUCCESS) {
			//error condition - check atdebug.log
		}
		else {
//			wcout << L"Successfully initialized" << endl;
		}
	}
	/*i_retCode = AT_InitialiseUtilityLibrary();*/
	return 0;
}

DLLEXPORT int fin(AT_H Hndl) {
	AT_Close(Hndl);
	//wcout << L"Camera finalized" << endl;
	return 0;
}


DLLEXPORT int centreAOI(AT_H Hndl) {
	AT_64 width;
	AT_64 height;
	AT_64 widthMax = 2048;
	AT_64 heightMax = 2048;
	AT_64 Left;
	AT_64 Top;
	int binIndex;
	AT_64 scale;
	AT_GetInt(Hndl, L"AOIWidth", &width);
	AT_GetInt(Hndl, L"AOIHeight", &height);
//	AT_GetIntMax(Hndl, L"AOIHeight", &heightMax);
//	AT_GetIntMax(Hndl, L"AOIWidth", &widthMax);
	AT_GetEnumIndex(Hndl, L"AOIBinning", &binIndex);
	if (binIndex==4)
	{
		scale = 8;
	}
	else
	{
		scale = binIndex + 1;
	}
	Left = (widthMax - width * scale) / 2;
	Top = (heightMax - height * scale) / 2;
	AT_SetInt(Hndl, L"AOILeft", Left);
	AT_SetInt(Hndl, L"AOITop", Top);
	//cout << scale << endl;
	//cout << widthMax << ", " << width << endl;
	//cout << Left << ", " << Top << endl;
	return 0;
}

DLLEXPORT int setInitialSettings(AT_H Hndl) {
	int i_retCode;
	i_retCode = AT_SetEnumString(Hndl, L"Pixel Encoding", L"Mono12Packed");
	i_retCode = AT_SetEnumString(Hndl, L"Pixel Readout Rate", L"280 MHz");
	i_retCode = AT_SetEnumString(Hndl, L"SimplePreAmpGainControl", L"12-bit (low noise)");
	i_retCode = AT_SetEnumString(Hndl, L"TriggerMode", L"Internal");
	return 0;
}

int makeDirNamedTime(string dir_exist, string &dir) {
	time_t t = time(nullptr);
	struct tm lt;
	//convert format
	localtime_s(&lt, &t);

	//make original format
	std::stringstream s;
	s << dir_exist;
	s << "\\";
	s << std::setfill('0') << std::right << setw(2) << lt.tm_hour;
	s << "_";
	s << std::setfill('0') << std::right << setw(2) << lt.tm_min;
	s << "_";
	s << std::setfill('0') << std::right << setw(2) << lt.tm_sec;

	dir = s.str();

	_mkdir(dir.c_str());
	return 0;
}

DLLEXPORT int startFixedAcquisition(AT_H Hndl, const char* dir_save_p, int NumberOfFrames, int& counter) {
	AT_64 ImageSizeBytes;
	AT_GetInt(Hndl, L"Image Size Bytes", &ImageSizeBytes);
	int BufferSize = static_cast<int>(ImageSizeBytes);
	int NumberOfBuffers = 10;
	//Allocate a memory buffer to store one frame
	unsigned char** UserBuffers = new unsigned char*[NumberOfBuffers];
	//Pass this buffer to the SDK
	for (int i = 0; i < NumberOfBuffers; i++) {
		UserBuffers[i] = new unsigned char[BufferSize];
		AT_QueueBuffer(Hndl, UserBuffers[i], BufferSize);
	}
	AT_64 ImageHeight;
	AT_GetInt(Hndl, L"AOI Height", &ImageHeight);
	AT_64 ImageWidth;
	AT_GetInt(Hndl, L"AOI Width", &ImageWidth);
	AT_64 ImageStride;
	AT_GetInt(Hndl, L"AOI Stride", &ImageStride);
	double FrameRate;
	AT_GetFloat(Hndl, L"FrameRate", &FrameRate);
	//cout << "ImageHeight = " << ImageHeight << endl;
	//cout << "ImageWidth = " << ImageWidth << endl;
	AT_SetEnumString(Hndl, L"CycleMode", L"Continuous");
	// make directory
	string dirname;
	string dir_save = string(dir_save_p);
	makeDirNamedTime(dir_save, dirname);
	string filename;
	// make metadata text file
	ofstream ometa(dirname + "\\metaSpool.txt");
	if (!ometa) {
		cerr << "Failed to save metadata" << endl;
	}
	else {
		ometa << ImageSizeBytes << endl;
		ometa << "mono12packed" << endl;
		ometa << ImageStride << endl;
		ometa << ImageHeight << endl;
		ometa << ImageWidth << endl;
		ometa << FrameRate << endl;
		ometa.close();
	}
	//Start the Acquisition running
	AT_Command(Hndl, L"Acquisition Start");

	cout << "Waiting for acquisition ..." << endl << endl;
	//Sleep in this thread until data is ready, in this case set
	//the timeout to infinite for simplicity
	unsigned char* Buffer;
	unsigned short* ImgArry = new unsigned short[ImageHeight * ImageWidth];
	for (int i = 0; i < NumberOfFrames; i++) {
		if (AT_WaitBuffer(Hndl, &Buffer, &BufferSize, 1000) == AT_SUCCESS) {
			std::stringstream s;
			s << dirname;
			s << "\\spool_";
			s << std::setfill('0') << std::right << setw(6) << i;
			s << ".dat";

			filename = s.str();
			ofstream ofs(filename, ios_base::out | ios_base::binary);
			if (!ofs) {
				cerr << "Failed to save" << endl;
			}
			else {
				ofs.write((const char*)Buffer, BufferSize);
				ofs.close();
			}
			//Requeue the buffer
			AT_QueueBuffer(Hndl, Buffer, BufferSize);
			counter = i + 1;
		}
		else {
			cout << "Timeout occurred check the log file ..." << endl << endl;
			break;
		}
	}

	for (int i = 0; i < NumberOfBuffers; i++) {
		delete[] UserBuffers[i];
	}
	delete[] UserBuffers;
	delete[] ImgArry;
	//Stop the Acquisition
	AT_Command(Hndl, L"Acquisition Stop");
	AT_Flush(Hndl);
	return 0;
}

DLLEXPORT int startFixedAcquisitionFile(AT_H Hndl, const char* dir_save_p, int NumberOfFrames, int& counter, bool& Stop) {
	AT_64 ImageSizeBytes;
	AT_GetInt(Hndl, L"Image Size Bytes", &ImageSizeBytes);
	int BufferSize = static_cast<int>(ImageSizeBytes);
	int NumberOfBuffers = 10;
	//Allocate a memory buffer to store one frame
	unsigned char** UserBuffers = new unsigned char* [NumberOfBuffers];
	//Pass this buffer to the SDK
	for (int i = 0; i < NumberOfBuffers; i++) {
		UserBuffers[i] = new unsigned char[BufferSize];
		AT_QueueBuffer(Hndl, UserBuffers[i], BufferSize);
	}
	AT_64 ImageHeight;
	AT_GetInt(Hndl, L"AOI Height", &ImageHeight);
	AT_64 ImageWidth;
	AT_GetInt(Hndl, L"AOI Width", &ImageWidth);
	AT_64 ImageStride;
	AT_GetInt(Hndl, L"AOI Stride", &ImageStride);
	double FrameRate;
	AT_GetFloat(Hndl, L"FrameRate", &FrameRate);
	//cout << "ImageHeight = " << ImageHeight << endl;
	//cout << "ImageWidth = " << ImageWidth << endl;
	AT_SetEnumString(Hndl, L"CycleMode", L"Continuous");
	// make directory
	string dirname;
	string dir_save = string(dir_save_p);
	makeDirNamedTime(dir_save, dirname);
	const char* filename;
	// make metadata text file
	ofstream ometa(dirname + "\\metaSpool.txt");
	if (!ometa) {
		cerr << "Failed to save metadata" << endl;
	}
	else {
		ometa << ImageSizeBytes << endl;
		ometa << "mono12packed" << endl;
		ometa << ImageStride << endl;
		ometa << ImageHeight << endl;
		ometa << ImageWidth << endl;
		ometa << FrameRate << endl;
		//Start the Acquisition running
		AT_Command(Hndl, L"Acquisition Start");

		cout << "Waiting for acquisition ..." << endl << endl;
		//Sleep in this thread until data is ready, in this case set
		//the timeout to infinite for simplicity
		unsigned char* Buffer;
		unsigned short* ImgArry = new unsigned short[ImageHeight * ImageWidth];
		std::stringstream s;
		s << dirname;
		s << "\\spool.dat";

		string temp = s.str();
		filename = temp.c_str();
		FILE* fpw;
		if (fopen_s(&fpw, filename, "wb")) {
			cerr << "Failed to save" << endl;
		}
		else {
			for (int i = 0; i < NumberOfFrames; i++) {
				if (Stop) {
					break;
				}
				if (AT_WaitBuffer(Hndl, &Buffer, &BufferSize, 1000) == AT_SUCCESS) {
					fwrite(Buffer, 1, BufferSize, fpw);
					//Requeue the buffer
					AT_QueueBuffer(Hndl, Buffer, BufferSize);
					counter = i + 1;
				}
				else {
					cout << "Timeout occurred" << endl << endl;
					break;
				}
			}
			fclose(fpw);
			ometa << counter;
			ometa.close();
		}
		delete[] ImgArry;
	}

	for (int i = 0; i < NumberOfBuffers; i++) {
		delete[] UserBuffers[i];
	}
	delete[] UserBuffers;

	//Stop the Acquisition
	AT_Command(Hndl, L"Acquisition Stop");
	AT_Flush(Hndl);
	return 0;
}

DLLEXPORT int startFixedAcquisitionFilePiezo(AT_H Hndl, const char* dir_save_p, int NumberOfFrames, int& counter, bool& Stop,  int ID) {
	AT_64 ImageSizeBytes;
	AT_GetInt(Hndl, L"Image Size Bytes", &ImageSizeBytes);
	int BufferSize = static_cast<int>(ImageSizeBytes);
	int NumberOfBuffers = 10;
	//Allocate a memory buffer to store one frame
	unsigned char** UserBuffers = new unsigned char* [NumberOfBuffers];
	//Pass this buffer to the SDK
	for (int i = 0; i < NumberOfBuffers; i++) {
		UserBuffers[i] = new unsigned char[BufferSize];
		AT_QueueBuffer(Hndl, UserBuffers[i], BufferSize);
	}
	AT_64 ImageHeight;
	AT_GetInt(Hndl, L"AOI Height", &ImageHeight);
	AT_64 ImageWidth;
	AT_GetInt(Hndl, L"AOI Width", &ImageWidth);
	AT_64 ImageStride;
	AT_GetInt(Hndl, L"AOI Stride", &ImageStride);
	double FrameRate;
	AT_GetFloat(Hndl, L"FrameRate", &FrameRate);
	//cout << "ImageHeight = " << ImageHeight << endl;
	//cout << "ImageWidth = " << ImageWidth << endl;
	AT_SetEnumString(Hndl, L"CycleMode", L"Continuous");
	// make directory
	string dirname;
	string dir_save = string(dir_save_p);
	makeDirNamedTime(dir_save, dirname);
	const char* filename;
	// make metadata text file
	ofstream ometa(dirname + "\\metaSpool.txt");
	if (!ometa) {
		cerr << "Failed to save metadata" << endl;
	}
	else {
		ometa << ImageSizeBytes << endl;
		ometa << "mono12packed" << endl;
		ometa << ImageStride << endl;
		ometa << ImageHeight << endl;
		ometa << ImageWidth << endl;
		ometa << FrameRate << endl;
		double* pos = new double[NumberOfFrames];
		//Start the Acquisition running
		AT_Command(Hndl, L"Acquisition Start");

		cout << "Waiting for acquisition ..." << endl << endl;
		//Sleep in this thread until data is ready, in this case set
		//the timeout to infinite for simplicity
		unsigned char* Buffer;
		unsigned short* ImgArry = new unsigned short[ImageHeight * ImageWidth];
		std::stringstream s;
		s << dirname;
		s << "\\spool.dat";

		string temp = s.str();
		filename = temp.c_str();
		FILE* fpw;
		if (fopen_s(&fpw, filename, "wb")) {
			cerr << "Failed to save" << endl;
		}
		else {
			for (int i = 0; i < NumberOfFrames; i++) {
				if (Stop) {
					break;
				}
				if (AT_WaitBuffer(Hndl, &Buffer, &BufferSize, 1000) == AT_SUCCESS) {
          getPiezoPosition(ID, pos[i]);
					fwrite(Buffer, 1, BufferSize, fpw);
					//Requeue the buffer
					AT_QueueBuffer(Hndl, Buffer, BufferSize);
					counter = i + 1;
				}
				else {
					cout << "Timeout occurred" << endl << endl;
					break;
				}
			}
			fclose(fpw);
			ometa << counter;
			ometa.close();
		}
		delete[] ImgArry;
		ofstream oz(dirname + "\\zPosition.csv");
		if (!oz) {
			cerr << "Failed to save zpositions" << endl;
		}
		else {
			for (int i = 0; i < counter; i++)
			{
				oz << pos[i] << endl;
			}
			oz.close();
		}
		delete[] pos;
	}
	for (int i = 0; i < NumberOfBuffers; i++) {
		delete[] UserBuffers[i];
	}
	delete[] UserBuffers;

	//Stop the Acquisition
	AT_Command(Hndl, L"Acquisition Stop");
	AT_Flush(Hndl);
	return 0;
}

DLLEXPORT int startFixedAcquisitionPiezo(AT_H Hndl, const char* dir_save_p, int NumberOfFrames, int& counter, int ID) {
	AT_64 ImageSizeBytes;
	AT_GetInt(Hndl, L"Image Size Bytes", &ImageSizeBytes);
	int BufferSize = static_cast<int>(ImageSizeBytes);
	int NumberOfBuffers = 10;
	//Allocate a memory buffer to store one frame
	unsigned char** UserBuffers = new unsigned char* [NumberOfBuffers];
	//Pass this buffer to the SDK
	for (int i = 0; i < NumberOfBuffers; i++) {
		UserBuffers[i] = new unsigned char[BufferSize];
		AT_QueueBuffer(Hndl, UserBuffers[i], BufferSize);
	}
	AT_64 ImageHeight;
	AT_GetInt(Hndl, L"AOI Height", &ImageHeight);
	AT_64 ImageWidth;
	AT_GetInt(Hndl, L"AOI Width", &ImageWidth);
	AT_64 ImageStride;
	AT_GetInt(Hndl, L"AOI Stride", &ImageStride);
	//cout << "ImageHeight = " << ImageHeight << endl;
	//cout << "ImageWidth = " << ImageWidth << endl;
	AT_SetEnumString(Hndl, L"CycleMode", L"Continuous");
	// make directory
	string dirname;
	string dir_save = string(dir_save_p);
	makeDirNamedTime(dir_save, dirname);
	string filename;
	// make metadata text file
	ofstream ometa(dirname + "\\metaSpool.txt");
	if (!ometa) {
		cerr << "Failed to save metadata" << endl;
	}
	else {
		ometa << ImageSizeBytes << endl;
		ometa << "mono12packed" << endl;
		ometa << ImageStride << endl;
		ometa << ImageHeight << endl;
		ometa << ImageWidth << endl;
		ometa.close();
	}
	double* pos = new double[NumberOfFrames];
	//Start the Acquisition running
	AT_Command(Hndl, L"Acquisition Start");

	cout << "Waiting for acquisition ..." << endl << endl;
	//Sleep in this thread until data is ready, in this case set
	//the timeout to infinite for simplicity
	unsigned char* Buffer;
	unsigned short* ImgArry = new unsigned short[ImageHeight * ImageWidth];
	for (int i = 0; i < NumberOfFrames; i++) {
		if (AT_WaitBuffer(Hndl, &Buffer, &BufferSize, 1000) == AT_SUCCESS) {
			getPiezoPosition(ID, pos[i]);
			std::stringstream s;
			s << dirname;
			s << "\\spool_";
			s << std::setfill('0') << std::right << setw(6) << i;
			s << ".dat";

			filename = s.str();
			ofstream ofs(filename, ios_base::out | ios_base::binary);
			if (!ofs) {
				cerr << "Failed to save" << endl;
			}
			else {
				ofs.write((const char*)Buffer, BufferSize);
				ofs.close();
			}
			//Requeue the buffer
			AT_QueueBuffer(Hndl, Buffer, BufferSize);
			counter = i + 1;
		}
		else {
			cout << "Timeout occurred check the log file ..." << endl << endl;
			break;
		}
	}

	for (int i = 0; i < NumberOfBuffers; i++) {
		delete[] UserBuffers[i];
	}
	ofstream oz(dirname + "\\zPosition.csv");
	if (!oz) {
		cerr << "Failed to save zpositions" << endl;
	}
	else {
		for (int i = 0; i < NumberOfFrames; i++)
		{
			oz << pos[i] << endl;
		}
		oz.close();
	}
	delete[] pos;
	delete[] UserBuffers;
	delete[] ImgArry;
	//Stop the Acquisition
	AT_Command(Hndl, L"Acquisition Stop");
	AT_Flush(Hndl);
	return 0;
}

DLLEXPORT int prepareBuffers(AT_H Handle) {
	AT_64 ImageSizeBytes;
	int ret = AT_GetInt(Handle, L"Image Size Bytes", &ImageSizeBytes);
	int BufferSize = static_cast<int>(ImageSizeBytes);
	int NumberOfBuffers = 10;
	//Allocate a memory buffer to store one frame
	unsigned char** UserBuffers = new unsigned char*[NumberOfBuffers];
	//Pass this buffer to the SDK
	for (int i = 0; i < NumberOfBuffers; i++) {
		UserBuffers[i] = new unsigned char[BufferSize];
		AT_QueueBuffer(Handle, UserBuffers[i], BufferSize);
	}
	return 0;
}

DLLEXPORT int convertBuffer(AT_U8* inputBuffer, unsigned short* outputBuffer, AT_64 width, AT_64 height, AT_64 stride) {
	return AT_ConvertBuffer(inputBuffer, reinterpret_cast<unsigned char*>(outputBuffer), width, height, stride, L"Mono12Packed", L"Mono16");
}

inline bool greaterArea(const vector<cv::Point>& cLeft, const vector<cv::Point>& cRight) {
	return cv::contourArea(cLeft) > cv::contourArea(cRight);
}

struct nsPrms
{
	bool norm;
	double normMax;
	double normMin;
	double sigma;
};

struct blurPrms
{
	bool exe;
	int sizeX;
	int sizeY;
	double sigma;
};

struct thresPrms
{
	bool exe;
	double thres;
	double maxVal;
	int type;
};

struct contPrms
{
	bool exe;
	int num;
};

struct processPrms
{
	nsPrms ns;
	blurPrms blur;
	thresPrms thres;
	contPrms cont;
};


inline int processImage(queue<cv::Point2f>& que_p, AT_64 ImageHeight, AT_64 ImageWidth, unsigned short* ImgArry, processPrms prms) {
	cv::Mat blurImg, normImg, binImg, mean, std;
	double meanVal, stdVal, alpha;
	vector<vector<cv::Point>> contours;
	//LARGE_INTEGER time1, time2, time3, time4, time5, freq;
	//QueryPerformanceFrequency(&freq);
	//QueryPerformanceCounter(&time1);
	cv::Mat m1(ImageHeight, ImageWidth, CV_16U, ImgArry);
	if (prms.ns.norm) {
		alpha = 256 / (prms.ns.normMax - prms.ns.normMin);
		m1.convertTo(normImg, CV_8U, alpha, -prms.ns.normMin * alpha);
	}
	else {
		cv::meanStdDev(m1, mean, std);
		meanVal = mean.at<double>(0);
		stdVal = std.at<double>(0);
		alpha = prms.ns.sigma / stdVal;
		m1.convertTo(normImg, CV_8U, alpha, (double)128 - meanVal * alpha);
	}
	//QueryPerformanceCounter(&time2);
	if (prms.blur.exe) {
		GaussianBlur(normImg, blurImg, cv::Size(prms.blur.sizeX, prms.blur.sizeY), prms.blur.sigma);
	}
	else {
		blurImg = normImg;
	}
	//QueryPerformanceCounter(&time3);
	if (prms.thres.exe) {
		cv::threshold(blurImg, binImg, prms.thres.thres, 255, prms.thres.type);
	}
	else {
		binImg = blurImg;
	}
	//QueryPerformanceCounter(&time4);
	if (prms.cont.exe) {
		cv::findContours(binImg, contours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
		int num = prms.cont.num;
		float x = 0.0;
		float y = 0.0;
		cv::Moments mu;
		if (contours.size() < num) {
			que_p.push(cv::Point2f(0.0, 0.0));
		}
		else {
			sort(contours.begin(), contours.end(), greaterArea);
			for (size_t i = 0; i < num; i++)
			{
				mu = cv::moments(contours[i]);
				x += mu.m10 / mu.m00;
				y += mu.m01 / mu.m00;
			}
			cv::Point2f mc = cv::Point2f(x/num, y/num);
			que_p.push(mc);
		}
	}
	else {
		que_p.push(cv::Point2f(0.0, 0.0));
	}
	//QueryPerformanceCounter(&time5);
	//cout << (double)(time2.QuadPart - time1.QuadPart) / freq.QuadPart << ", " << (double)(time3.QuadPart - time2.QuadPart) / freq.QuadPart << ", " << (double)(time4.QuadPart - time3.QuadPart) / freq.QuadPart << ", " << (double)(time5.QuadPart - time4.QuadPart) / freq.QuadPart << endl;
}

DLLEXPORT int processImage(float* point, AT_64 ImageHeight, AT_64 ImageWidth, unsigned short* ImgArry, processPrms prms) {
	cv::Mat blurImg, normImg, binImg, mean, std;
	double meanVal, stdVal, alpha;
	vector<vector<cv::Point>> contours;
	cv::Moments mu;
	float x = 0.0;
	float y = 0.0;
	//LARGE_INTEGER time1, time2, time3, time4, time5, freq;
	//QueryPerformanceFrequency(&freq);
	//QueryPerformanceCounter(&time1);
	cv::Mat m1(ImageHeight, ImageWidth, CV_16U, ImgArry);
	if (prms.ns.norm) {
		alpha = 256 / (prms.ns.normMax - prms.ns.normMin);
		m1.convertTo(normImg, CV_8U, alpha, -prms.ns.normMin * alpha);
	}
	else {
		cv::meanStdDev(m1, mean, std);
		meanVal = mean.at<double>(0);
		stdVal = std.at<double>(0);
		alpha = prms.ns.sigma / stdVal;
		m1.convertTo(normImg, CV_8U, alpha, (double)128 - meanVal * alpha);
	}
	//QueryPerformanceCounter(&time2);
	if (prms.blur.exe) {
		GaussianBlur(normImg, blurImg, cv::Size(prms.blur.sizeX, prms.blur.sizeY), prms.blur.sigma);
	}
	else {
		blurImg = normImg;
	}
	//QueryPerformanceCounter(&time3);
	if (prms.thres.exe) {
		cv::threshold(blurImg, binImg, prms.thres.thres, prms.thres.maxVal, prms.thres.type);
	}
	else {
		binImg = blurImg;
	}
	//QueryPerformanceCounter(&time4);
	if (prms.cont.exe) {
		cv::findContours(binImg, contours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
		if (contours.size() < prms.cont.num) {
			return 1;
		}
		else {
			sort(contours.begin(), contours.end(), greaterArea);
			for (size_t i = 0; i < prms.cont.num; i++)
			{
				mu = cv::moments(contours[i]);
				x += mu.m10 / mu.m00;
				y += mu.m01 / mu.m00;
			}
			cv::Point2f mc = cv::Point2f(x/prms.cont.num, y/prms.cont.num);
			point[0] = mc.x;
			point[1] = mc.y;
		}
	}
	//QueryPerformanceCounter(&time5);
	//cout << (double)(time2.QuadPart - time1.QuadPart) / freq.QuadPart << ", " << (double)(time3.QuadPart - time2.QuadPart) / freq.QuadPart << ", " << (double)(time4.QuadPart - time3.QuadPart) / freq.QuadPart << ", " << (double)(time5.QuadPart - time4.QuadPart) / freq.QuadPart << endl;
	return 0;
}


DLLEXPORT int processImageShow(AT_64 ImageHeight, AT_64 ImageWidth, unsigned short* ImgArry, processPrms prms,\
		unsigned char* outBuffer, double& maxVal, double& minVal) {
	cv::Mat blurImg, normImg, maxMat1, maxMat2, minMat1, minMat2, copyImg, mean, std;
	double alpha, meanVal, stdVal;
	vector<vector<cv::Point>> contours;
	vector<unsigned char> array;
	int num;
	cv::Mat m1(ImageHeight, ImageWidth, CV_16U, ImgArry);
	cv::Mat binImg(ImageHeight, ImageWidth, CV_8U, outBuffer);
	cv::minMaxLoc(m1, &minVal, &maxVal);
	if (prms.ns.norm) {
		alpha = 256 / (prms.ns.normMax - prms.ns.normMin);
		m1.convertTo(normImg, CV_8U, alpha, -prms.ns.normMin * alpha);
	}
	else {
		cv::meanStdDev(m1, mean, std);
		meanVal = mean.at<double>(0);
		stdVal = std.at<double>(0);
		alpha = prms.ns.sigma / stdVal;
		m1.convertTo(normImg, CV_8U, alpha, (double)128 - meanVal * alpha);
	}
	if (prms.blur.exe) {
		GaussianBlur(normImg, blurImg, cv::Size(prms.blur.sizeX, prms.blur.sizeY), prms.blur.sigma);
	}
	else {
		blurImg = normImg;
	}
	if (prms.thres.exe) {
		cv::threshold(blurImg, binImg, prms.thres.thres, prms.thres.maxVal, prms.thres.type);
		if (prms.cont.exe) {
			num = prms.cont.num;
			binImg.copyTo(copyImg);
			cv::findContours(copyImg, contours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
			sort(contours.begin(), contours.end(), greaterArea);
			if (contours.size() < num)
			{
				num = contours.size();
			}
			for (size_t i = 0; i < num; i++)
			{
				cv::drawContours(binImg, contours, i, cv::Scalar(128, 128, 128), 3);
			}
		}
	}
	else {
		blurImg.copyTo(binImg);
	}
}

void processDatas(int number, queue<unsigned short*>& que, queue<cv::Point2f>& que_p, AT_64 ImageHeight, AT_64 ImageWidth, processPrms prms) {
	unsigned short* ImgArry;
	int i = 0;
	//LARGE_INTEGER freq, start, end, mid1, mid2;
	//QueryPerformanceFrequency(&freq);
	while (i < number){
		if (que.size()>0) {
			//cout << "size: " << que.size() << endl;
			//QueryPerformanceCounter(&start);
			ImgArry = que.front();
			//QueryPerformanceCounter(&mid1);
			processImage(que_p, ImageHeight, ImageWidth, ImgArry, prms);
			//QueryPerformanceCounter(&mid2);
			que.pop();
			//QueryPerformanceCounter(&end);
			//cout << "time: " << (double)(mid1.QuadPart - start.QuadPart) / freq.QuadPart << ", " << (double)(mid2.QuadPart - mid1.QuadPart) / freq.QuadPart << ", " << (double)(end.QuadPart - mid2.QuadPart) / freq.QuadPart << endl;
			i++;
		}
	}
}

void processPoints(int number, queue<cv::Point2f>& que_p, float* pnt, int mode, float p_X, float p_Y, HANDLE handle, float thres, string dirname){
	cv::Point2f point = cv::Point2f(0.0, 0.0);
	cv::Point2f p_front;
	bool* results = new bool[number];
	cv::Point2f* points = new cv::Point2f[number];
	int i = 0;
	int j = 0;
	float dx;
	float dy;
	LARGE_INTEGER freq, start, end;
	QueryPerformanceFrequency(&freq);
	if (mode==1)
	{
		while (i < number) {
			if (que_p.size() > 0) {
				p_front = que_p.front();
				que_p.pop();
				if (p_front == cv::Point2f(0.0, 0.0))
				{
					j++;
				}
				point = point + p_front;
				i++;
			}
		}
		if (i == j)
		{
			pnt[0] = 0.0;
			pnt[1] = 0.0;
		}
		else {
			point = point / (i-j);
			pnt[0] = point.x;
			pnt[1] = point.y;
		}
		ofstream ocntr(dirname + "\\center.csv");
		ocntr << point.x << ", " << point.y << endl;
		ocntr.close();
	}
	else if (mode==2)
	{
		float thres_2 = pow(thres, 2.0);
		while (i < number) {
			if (que_p.size() > 0) {
				QueryPerformanceCounter(&start);
				point = que_p.front();
				que_p.pop();
				dx = abs(point.x - p_X);
				dy = abs(point.y - p_Y);
				points[i] = point;
				if (pow(dx, 2.0) + pow(dy, 2.0) < thres_2)
				{
				#ifdef WM_RSDIO
					WritePort(handle, 1);
				#endif // WM_RSDIO
					results[i] = false;
				}
				else {
				#ifdef WM_RSDIO
					WritePort(handle, 0);
				#endif // WM_RSDIO
					results[i] = true;
				}
				QueryPerformanceCounter(&end);
				cout << "time: " << (double)(end.QuadPart - start.QuadPart) / freq.QuadPart << endl;
				i++;
			}
		}
#ifdef WM_RSDIO
		WritePort(handle, 1);
#endif // WM_RSDIO
		ofstream oresult(dirname + "\\result.csv");
		for (size_t i = 0; i < number; i++)
		{
			oresult << i << ", " << points[i].x << ", " << points[i].y << ", " << results[i] << endl;
		}
		oresult.close();
	}
	delete[] results;
	delete[] points;
}

DLLEXPORT int multithread(AT_H Hndl, const char* dir_save_p, int NumberOfFrames, int& counter, processPrms prms, float* point, float* center, float DIOthres, HANDLE DIOhandle, int mode) {
	AT_64 ImageSizeBytes;
	AT_64 ImageHeight;
	AT_64 ImageWidth;
	AT_64 ImageStride;
	AT_GetInt(Hndl, L"Image Size Bytes", &ImageSizeBytes);
	int BufferSize = static_cast<int>(ImageSizeBytes);
	int NumberOfBuffers = 30;
	//Allocate a memory buffer to store one frame
	unsigned char** UserBuffers = new unsigned char*[NumberOfBuffers];
	//Pass this buffer to the SDK
	for (int i = 0; i < NumberOfBuffers; i++) {
		UserBuffers[i] = new unsigned char[BufferSize];
		AT_QueueBuffer(Hndl, UserBuffers[i], BufferSize);
	}

	AT_GetInt(Hndl, L"AOI Height", &ImageHeight);
	AT_GetInt(Hndl, L"AOI Width", &ImageWidth);
	AT_GetInt(Hndl, L"AOI Stride", &ImageStride);
	//cout << "ImageHeight = " << ImageHeight << endl;
	//cout << "ImageWidth = " << ImageWidth << endl;

	unsigned char* Buffer;
	cv::Mat blurImg, normImg, binImg, maxMat1, maxMat2, minMat1, minMat2, copyImg, mean, std;
	//LARGE_INTEGER freq, start, mid, end;
	vector<vector<cv::Point>> contours;
	queue<unsigned short*> que;
	queue < cv::Point2f > quePoint;
	unsigned short** ImgArrys = new unsigned short*[NumberOfBuffers];
	for (size_t i = 0; i < NumberOfBuffers; i++)
	{
		ImgArrys[i] = new unsigned short[ImageHeight*ImageWidth];
	}
	AT_SetEnumString(Hndl, L"CycleMode", L"Continuous");
	// make directory
	string dirname;
	string filename;
	FILE *fpw;
	string dir_save = string(dir_save_p);
	makeDirNamedTime(dir_save, dirname);

	// make metadata text file
	ofstream ometa(dirname + "\\metaSpool.txt");
	if (!ometa) {
		cerr << "Failed to save metadata" << endl;
	}
	else {
		ometa << ImageSizeBytes << endl;
		ometa << "mono12packed" << endl;
		ometa << ImageStride << endl;
		ometa << ImageHeight << endl;
		ometa << ImageWidth << endl;
		ometa.close();
	}
	//Start the Acquisition running
	AT_Command(Hndl, L"Acquisition Start");

	cout << "Waiting for acquisition ..." << endl << endl;
	//Sleep in this thread until data is ready, in this case set
	//the timeout to infinite for simplicity

	thread th_process(processDatas, NumberOfFrames, ref(que), ref(quePoint), ImageHeight, ImageWidth, prms);
	thread th_process_2(processPoints, NumberOfFrames, ref(quePoint), point, mode, center[0], center[1], DIOhandle, DIOthres, dirname);

	//QueryPerformanceFrequency(&freq);
	for (int i = 0; i < NumberOfFrames; i++) {
		if (AT_WaitBuffer(Hndl, &Buffer, &BufferSize, 1000) == AT_SUCCESS) {
			//QueryPerformanceCounter(&start);
			int j = i % 30;
			AT_ConvertBuffer(Buffer, reinterpret_cast<unsigned char*>(ImgArrys[j]), ImageWidth, ImageHeight, ImageStride, L"Mono12Packed", L"Mono16");
			que.push(ImgArrys[j]);
			//QueryPerformanceCounter(&mid);
			std::stringstream s;
			s << dirname;
			s << "\\spool_";
			s << std::setfill('0') << std::right << setw(6) << i;
			s << ".dat";

			filename = s.str();

			fopen_s(&fpw, filename.c_str(), "wb");
			fwrite(Buffer, 1, BufferSize, fpw);
			fclose(fpw);
			//QueryPerformanceCounter(&end);
			//cout << "time: " << (double)(mid.QuadPart - start.QuadPart) / freq.QuadPart << endl;
			//cout << (double)(end.QuadPart - mid.QuadPart) / freq.QuadPart << endl;
			AT_QueueBuffer(Hndl, Buffer, BufferSize);

		}
	}
	th_process.join();
	th_process_2.join();
	AT_Command(Hndl, L"Acquisition Stop");
	AT_Flush(Hndl);
	for (int i = 0; i < NumberOfBuffers; i++)
	{
		delete[] UserBuffers[i];
		delete[] ImgArrys[i];
	}
	delete[] UserBuffers;
	delete[] ImgArrys;
}

//DLLEXPORT int analyzeDataSeries(const char* dir_p, int start, int end) {
//	cout << "CALLED" << endl;
//	string dir = string(dir_p);
//	string filename, str;
//	int imgSize;
//	AT_64 ImageWidth, ImageHeight, ImageStride;
//	//unsigned char* buffer = new unsigned char[imgSize];
//	ifstream ifs(dir + string("\\metaSpool.txt"));
//	getline(ifs, str);
//	sscanf_s(str.c_str(), "%d", &imgSize);
//	cout << imgSize << endl;
//	for (int i = start; i < end; i++)
//	{
//		ostringstream sout;
//		sout << setfill('0') << setw(6) << i;
//		filename = dir + string("\\spool_") + sout.str() + string(".dat");
//		cout << filename << endl;
//	}
//}


DLLEXPORT int InitialiseLibrary() {
	return AT_InitialiseLibrary();
}

DLLEXPORT int FinaliseLibrary() {
	return AT_FinaliseLibrary();
}

DLLEXPORT int Open(int DeviceIndex, AT_H* Handle) {
	return AT_Open(DeviceIndex, Handle);
}

DLLEXPORT int Close(AT_H Hndl) {
	return AT_Close(Hndl);
}

//typedef int(*FeatureCallback)(AT_H Hndl, AT_WC* Feature, void* Context);
//int AT_RegisterFeatureCallback(AT_H Hndl, AT_WC* Feature, FeatureCallback EvCallback,
//	void* Context);
//int AT_UnregisterFeatureCallback(AT_H Hndl, AT_WC* Feature, FeatureCallback EvCallback,
//	void* Context);

DLLEXPORT int IsImplemented(AT_H Hndl, AT_WC* Feature, AT_BOOL* Implemented) {
	return AT_IsImplemented(Hndl, Feature, Implemented);
}

DLLEXPORT int IsReadOnly(AT_H Hndl, AT_WC* Feature, AT_BOOL* ReadOnly) {
	return AT_IsReadOnly(Hndl, Feature, ReadOnly);
}

DLLEXPORT int IsReadable(AT_H Hndl, AT_WC* Feature, AT_BOOL* Readable) {
	return AT_IsReadable(Hndl, Feature, Readable);
}

DLLEXPORT int IsWritable(AT_H Hndl, AT_WC* Feature, AT_BOOL* Writable) {
	return AT_IsWritable(Hndl, Feature, Writable);
}

DLLEXPORT int SetInt(AT_H Hndl, AT_WC* Feature, AT_64 Value) {
	return AT_SetInt(Hndl, Feature, Value);
}

DLLEXPORT int GetInt(AT_H Hndl, AT_WC* Feature, AT_64* Value) {
	return AT_GetInt(Hndl, Feature, Value);
}

DLLEXPORT int GetIntMax(AT_H Hndl, AT_WC* Feature, AT_64* MaxValue) {
	return AT_GetIntMax(Hndl, Feature, MaxValue);
}

DLLEXPORT int GetIntMin(AT_H Hndl, AT_WC* Feature, AT_64* MinValue) {
	return AT_GetIntMin(Hndl, Feature, MinValue);
}

DLLEXPORT int SetFloat(AT_H Hndl, AT_WC* Feature, double Value) {
	return AT_SetFloat(Hndl, Feature, Value);
}

DLLEXPORT int GetFloat(AT_H Hndl, AT_WC* Feature, double* Value) {
	return AT_GetFloat(Hndl, Feature, Value);
}

DLLEXPORT int GetFloatMax(AT_H Hndl, AT_WC* Feature, double* MaxValue) {
	return AT_GetFloatMax(Hndl, Feature, MaxValue);
}

DLLEXPORT int GetFloatMin(AT_H Hndl, AT_WC* Feature, double* MinValue) {
	return AT_GetFloatMin(Hndl, Feature, MinValue);
}

DLLEXPORT int SetBool(AT_H Hndl, AT_WC* Feature, AT_BOOL Value) {
	return AT_SetBool(Hndl, Feature, Value);
}

DLLEXPORT int GetBool(AT_H Hndl, AT_WC* Feature, AT_BOOL* Value) {
	return AT_GetBool(Hndl, Feature, Value);
}

DLLEXPORT int SetEnumIndex(AT_H Hndl, AT_WC* Feature, int Value) {
	return AT_SetEnumIndex(Hndl, Feature, Value);
}

DLLEXPORT int SetEnumString(AT_H Hndl, AT_WC* Feature, AT_WC* String) {
	return AT_SetEnumString(Hndl, Feature, String);
}

DLLEXPORT int GetEnumIndex(AT_H Hndl, AT_WC* Feature, int* Value) {
	return AT_GetEnumIndex(Hndl, Feature, Value);
}

DLLEXPORT int GetEnumCount(AT_H Hndl, AT_WC* Feature, int* Count) {
	return AT_GetEnumCount(Hndl, Feature, Count);
}

DLLEXPORT int IsEnumIndexAvailable(AT_H Hndl, AT_WC* Feature, int Index, AT_BOOL* Available) {
	return AT_IsEnumIndexAvailable(Hndl, Feature, Index, Available);
}

DLLEXPORT int IsEnumIndexImplemented(AT_H Hndl, AT_WC* Feature, int Index, AT_BOOL* Implemented) {
	return AT_IsEnumIndexImplemented(Hndl, Feature, Index, Implemented);
}

DLLEXPORT int GetEnumStringByIndex(AT_H Hndl, AT_WC* Feature, int Index, AT_WC* String, int StringLength) {
	return AT_GetEnumStringByIndex(Hndl, Feature, Index, String, StringLength);
}

DLLEXPORT int Command(AT_H Hndl, AT_WC* Feature) {
	return AT_Command(Hndl, Feature);
}

DLLEXPORT int SetString(AT_H Hndl, AT_WC* Feature, AT_WC* Value) {
	return AT_SetString(Hndl, Feature, Value);
}

DLLEXPORT int GetString(AT_H Hndl, AT_WC* Feature, AT_WC* Value, int StringLength) {
	return AT_GetString(Hndl, Feature, Value, StringLength);
}

DLLEXPORT int GetStringMaxLength(AT_H Hndl, AT_WC* Feature, int* MaxStringLength) {
	return AT_GetStringMaxLength(Hndl, Feature, MaxStringLength);
}

DLLEXPORT int QueueBuffer(AT_H Hndl, AT_U8* Ptr, int PtrSize) {
	return AT_QueueBuffer(Hndl, Ptr, PtrSize);
}

DLLEXPORT int WaitBuffer(AT_H Hndl, AT_U8** Ptr, int* PtrSize, unsigned int Timeout) {
	return AT_WaitBuffer(Hndl, Ptr, PtrSize, Timeout);
}

DLLEXPORT int Flush(AT_H Hndl) {
	return AT_Flush(Hndl);
}

DLLEXPORT int InitialiseUtilityLibrary() {
	return AT_InitialiseUtilityLibrary();
}

DLLEXPORT int FinaliseUtilityLibrary() {
	return AT_FinaliseUtilityLibrary();
}
