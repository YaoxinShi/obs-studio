/*****************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or
nondisclosure agreement with Intel Corporation and may not be copied
or disclosed except in accordance with the terms of that agreement.
Copyright(c) 2005-2014 Intel Corporation. All Rights Reserved.

*****************************************************************************/

#ifndef __COMMON_UTILS_H__
#define __COMMON_UTILS_H__

#include <stdio.h>
#include <string>
#include <tchar.h>
#include <sstream>

#include "mfxvideo++.h"

// =================================================================
// OS-specific definitions of types, macro, etc...
// The following should be defined:
//  - mfxTime
//  - MSDK_FOPEN
//  - MSDK_SLEEP
#if defined(_WIN32) || defined(_WIN64)
#include "bits/windows_defs.h"
#elif defined(__linux__)
#include "bits/linux_defs.h"
#endif

#ifdef __cplusplus
typedef std::basic_string<TCHAR> msdk_tstring;
#endif
typedef TCHAR msdk_char;

#define msdk_printf _tprintf

#ifdef UNICODE
#define msdk_cout std::wcout
#define msdk_err std::wcerr
#else
#define msdk_cout std::cout
#define msdk_err std::cerr
#endif

typedef std::basic_string<msdk_char> msdk_string;
typedef std::basic_stringstream<msdk_char> msdk_stringstream;
typedef std::basic_ostream<msdk_char, std::char_traits<msdk_char> > msdk_ostream;
typedef std::basic_istream<msdk_char, std::char_traits<msdk_char> > msdk_istream;
typedef std::basic_fstream<msdk_char, std::char_traits<msdk_char> > msdk_fstream;

#if defined(_UNICODE)
#define MSDK_MAKE_BYTE_STRING(src,dest) \
    {\
        std::wstring wstr(src);\
        std::string str(wstr.begin(), wstr.end());\
        strcpy_s(dest, str.c_str());\
    }
#else
#define MSDK_MAKE_BYTE_STRING(src,dest) msdk_strcopy(dest, src);
#endif

// =================================================================
// Helper macro definitions...
#define MSDK_PRINT_RET_MSG(ERR)         {PrintErrString(ERR, __FILE__, __LINE__);}
#define MSDK_CHECK_RESULT(P, X, ERR)    {if ((X) > (P)) {MSDK_PRINT_RET_MSG(ERR); return ERR;}}
#define MSDK_CHECK_POINTER(P, ERR)      {if (!(P)) {MSDK_PRINT_RET_MSG(ERR); return ERR;}}
#define MSDK_CHECK_ERROR(P, X, ERR)     {if ((X) == (P)) {MSDK_PRINT_RET_MSG(ERR); return ERR;}}
#define MSDK_IGNORE_MFX_STS(P, X)       {if ((X) == (P)) {P = MFX_ERR_NONE;}}
#define MSDK_BREAK_ON_ERROR(P)          {if (MFX_ERR_NONE != (P)) break;}
#define MSDK_SAFE_DELETE_ARRAY(P)       {if (P) {delete[] P; P = NULL;}}
#define MSDK_ALIGN32(X)                 (((mfxU32)((X)+31)) & (~ (mfxU32)31))
#define MSDK_ALIGN16(value)             (((value + 15) >> 4) << 4)
#define MSDK_SAFE_RELEASE(X)            {if (X) { X->Release(); X = NULL; }}
#define MSDK_MAX(A, B)                  (((A) > (B)) ? (A) : (B))
#define MSDK_ZERO_MEMORY(VAR)           {memset(&VAR, 0, sizeof(VAR));}
#define MSDK_MEMCPY(dst, src, count) memcpy_s(dst, (count), (src), (count))

#define MSDK_STRING(x) _T(x)
#define MSDK_CHAR(x) _T(x)

enum MsdkTraceLevel {
	MSDK_TRACE_LEVEL_SILENT = -1,
	MSDK_TRACE_LEVEL_CRITICAL = 0,
	MSDK_TRACE_LEVEL_ERROR = 1,
	MSDK_TRACE_LEVEL_WARNING = 2,
	MSDK_TRACE_LEVEL_INFO = 3,
	MSDK_TRACE_LEVEL_DEBUG = 4,
};

#define MSDK_TRACE_LEVEL(level, ERR) if (level <= msdk_trace_get_level()) {msdk_err<<NoFullPath(MSDK_STRING(__FILE__)) << MSDK_STRING(" :")<< __LINE__ <<MSDK_STRING(" [") \
    <<level<<MSDK_STRING("] ") << ERR << std::endl;}

#define MSDK_TRACE_ERROR(ERR) MSDK_TRACE_LEVEL(MSDK_TRACE_LEVEL_ERROR, ERR)
#define MSDK_TRACE_INFO(ERR) MSDK_TRACE_LEVEL(MSDK_TRACE_LEVEL_INFO, ERR)

#define MSDK_MAX_FILENAME_LEN 1024

// Usage of the following two macros are only required for certain Windows DirectX11 use cases
#define WILL_READ  0x1000
#define WILL_WRITE 0x2000

// =================================================================
// Intel Media SDK memory allocator entrypoints....
// Implementation of this functions is OS/Memory type specific.
mfxStatus simple_alloc(mfxHDL pthis, mfxFrameAllocRequest* request, mfxFrameAllocResponse* response);
mfxStatus simple_lock(mfxHDL pthis, mfxMemId mid, mfxFrameData* ptr);
mfxStatus simple_unlock(mfxHDL pthis, mfxMemId mid, mfxFrameData* ptr);
mfxStatus simple_gethdl(mfxHDL pthis, mfxMemId mid, mfxHDL* handle);
mfxStatus simple_free(mfxHDL pthis, mfxFrameAllocResponse* response);




// LoadRawFrame: Reads raw frame from YUV file (YV12) into NV12 surface
// - YV12 is a more common format for YUV files than NV12 (therefore the conversion during read and write)
// - For the simulation case (fSource = NULL), the surface is filled with default image data
// LoadRawRGBFrame: Reads raw RGB32 frames from file into RGB32 surface
// - For the simulation case (fSource = NULL), the surface is filled with default image data

mfxStatus LoadRawFrame(mfxFrameSurface1* pSurface, FILE* fSource);
mfxStatus LoadRawRGBFrame(mfxFrameSurface1* pSurface, FILE* fSource);

// Write raw YUV (NV12) surface to YUV (YV12) file
mfxStatus WriteRawFrame(mfxFrameSurface1* pSurface, FILE* fSink);

// Write bit stream data for frame to file
mfxStatus WriteBitStreamFrame(mfxBitstream* pMfxBitstream, FILE* fSink);
// Read bit stream data from file. Stream is read as large chunks (= many frames)
mfxStatus ReadBitStreamData(mfxBitstream* pBS, FILE* fSource);

void ClearYUVSurfaceSysMem(mfxFrameSurface1* pSfc, mfxU16 width, mfxU16 height);
void ClearYUVSurfaceVMem(mfxMemId memId);
void ClearRGBSurfaceVMem(mfxMemId memId);

// Get free raw frame surface
int GetFreeSurfaceIndex(mfxFrameSurface1** pSurfacesPool, mfxU16 nPoolSize);

// For use with asynchronous task management
typedef struct {
    mfxBitstream mfxBS;
    mfxSyncPoint syncp;
} Task;

// Get free task
int GetFreeTaskIndex(Task* pTaskPool, mfxU16 nPoolSize);

// Initialize Intel Media SDK Session, device/display and memory manager
mfxStatus Initialize(mfxIMPL impl, mfxVersion ver, MFXVideoSession* pSession, mfxFrameAllocator* pmfxAllocator, mfxHDL *deviceHandle = NULL, bool bCreateSharedHandles = false, bool dx9hack = false);

// Release resources (device/display)
void Release();

// Convert frame type to string
char mfxFrameTypeString(mfxU16 FrameType);

void mfxGetTime(mfxTime* timestamp);

//void mfxInitTime();  might need this for Windows
double TimeDiffMsec(mfxTime tfinish, mfxTime tstart);


// =================================================================
// Utility functions used for loading MSDK plugins
msdk_string NoFullPath(const msdk_string &);
int  msdk_trace_get_level();
void msdk_trace_set_level(int);
bool msdk_trace_is_printable(int);

msdk_ostream & operator <<(msdk_ostream & os, MsdkTraceLevel tt);

template<typename T>
mfxStatus msdk_opt_read(const msdk_char* string, T& value);

template<size_t S>
mfxStatus msdk_opt_read(const msdk_char* string, msdk_char(&value)[S])
{
	value[0] = 0;
#if defined(_WIN32) || defined(_WIN64)
	value[S - 1] = 0;
	return (0 == _tcsncpy_s(value, string, S - 1)) ? MFX_ERR_NONE : MFX_ERR_UNKNOWN;
#else
	if (strlen(string) < S) {
		strncpy(value, string, S - 1);
		value[S - 1] = 0;
		return MFX_ERR_NONE;
	}
	return MFX_ERR_UNKNOWN;
#endif
}

template<typename T>
inline mfxStatus msdk_opt_read(const msdk_string& string, T& value)
{
	return msdk_opt_read(string.c_str(), value);
}

// =================================================================
// Utility functions, not directly tied to Media SDK functionality
void PrintErrString(int err, const char* filestr, int line);

#endif // __COMMON_UTILS_H__
