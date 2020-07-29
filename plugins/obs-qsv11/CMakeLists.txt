project(obs-qsv11)

if(MSVC)
	set(obs-qsv11_PLATFORM_DEPS
		w32-pthreads)
endif()

include_directories(libmfx/include/msdk/include)
include_directories(libmfx/include)

# Find OpenCV components
find_package(OpenCV COMPONENTS REQUIRED)
message(STATUS "=== OpenCV library status: ===")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/include)
include_directories(${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/src/extension)
include_directories(${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/external/tbb/include)

if (WIN32)
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings
    endif()
endif()

set(obs-qsv11_libmfx_SOURCES
	libmfx/src/main.cpp
	libmfx/src/mfx_critical_section.cpp
	libmfx/src/mfx_dispatcher.cpp
	libmfx/src/mfx_dispatcher_log.cpp
	libmfx/src/mfx_dxva2_device.cpp
	libmfx/src/mfx_function_table.cpp
	libmfx/src/mfx_library_iterator.cpp
	libmfx/src/mfx_load_dll.cpp
	libmfx/src/mfx_load_plugin.cpp
	libmfx/src/mfx_plugin_hive.cpp
	libmfx/src/mfx_win_reg_key.cpp
	)

set(obs-qsv11_libmfx_HEADERS
	libmfx/include/msdk/include/mfxastructures.h
	libmfx/include/msdk/include/mfxaudio.h
	libmfx/include/msdk/include/mfxaudio++.h
	libmfx/include/msdk/include/mfxcommon.h
	libmfx/include/msdk/include/mfxdefs.h
	libmfx/include/msdk/include/mfxjpeg.h
	libmfx/include/msdk/include/mfxmvc.h
	libmfx/include/msdk/include/mfxplugin.h
	libmfx/include/msdk/include/mfxplugin++.h
	libmfx/include/msdk/include/mfxsession.h
	libmfx/include/msdk/include/mfxstructures.h
	libmfx/include/msdk/include/mfxvideo.h
	libmfx/include/msdk/include/mfxvideo++.h
	libmfx/include/msdk/include/mfxvstructures.h
	libmfx/include/mfx_critical_section.h
	libmfx/include/mfx_dispatcher.h
	libmfx/include/mfx_dispatcher_defs.h
	libmfx/include/mfx_dispatcher_log.h
	libmfx/include/mfx_dxva2_device.h
	libmfx/include/mfx_exposed_functions_list.h
	libmfx/include/mfx_library_iterator.h
	libmfx/include/mfx_load_dll.h
	libmfx/include/mfx_load_plugin.h
	libmfx/include/mfx_plugin_hive.h
	libmfx/include/mfx_vector.h
	libmfx/include/mfx_win_reg_key.h
	libmfx/include/mfxaudio_exposed_functions_list.h
	)

set(text_SOURCES
	text/cnn.cpp
	text/image_grabber.cpp
	text/main_text.cpp
	text/main_segmentation.cpp
	text/main_sod.cpp
	text/text_detection.cpp
	text/text_recognition.cpp)

set(text_HEADERS
	text/cnn.hpp
	text/image_grabber.hpp
	text/main_text.h
	text/text_detection.hpp
	text/text_recognition.hpp)

set(obs-qsv11_SOURCES
	common_directx9.cpp
	common_directx11.cpp
	common_utils.cpp
	common_utils_windows.cpp
	device_directx9.cpp
	QSV_Encoder.cpp
	QSV_Encoder_Internal.cpp
	obs-qsv11.c
	obs-qsv11-plugin-main.c)

set(obs-qsv11_HEADERS
	bits/linux_defs.h
	bits/windows_defs.h
	common_directx9.h
	common_directx11.h
	device_directx9.h
	common_utils.h
	QSV_Encoder.h
	QSV_Encoder_Internal.h)

add_library(obs-qsv11 MODULE
	${obs-qsv11_SOURCES}
	${obs-qsv11_HEADERS}
	${obs-qsv11_libmfx_SOURCES}
	${obs-qsv11_libmfx_HEADERS}
	${text_SOURCES}
	${text_HEADERS}
	)
target_link_libraries(obs-qsv11
	libobs
	d3d9
	d3d11
	dxva2
	dxgi
	${obs-qsv11_PLATFORM_DEPS}
	${OpenCV_LIBS}
    debug ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Debug/inference_engined.lib
    debug ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Debug/inference_engine_c_apid.lib
	debug ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Debug/inference_engine_ir_readerd.lib
	debug ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Debug/inference_engine_legacyd.lib
	debug ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Debug/inference_engine_lp_transformationsd.lib
	debug ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Debug/inference_engine_onnx_readerd.lib
	debug ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Debug/inference_engine_preprocd.lib
	debug ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Debug/inference_engine_transformationsd.lib
    optimized ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Release/inference_engine.lib
	optimized ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Release/inference_engine_c_api.lib
	optimized ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Release/inference_engine_ir_reader.lib
	optimized ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Release/inference_engine_legacy.lib
	optimized ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Release/inference_engine_lp_transformations.lib
	optimized ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Release/inference_engine_onnx_reader.lib
	optimized ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Release/inference_engine_preproc.lib
	optimized ${OpenCV_INCLUDE_DIRS}/../../deployment_tools/inference_engine/lib/intel64/Release/inference_engine_transformations.lib
	)

target_compile_definitions(obs-qsv11 PRIVATE DX11_D3D)

source_group("obs-qsv11\\Source Files" FILES ${obs-qsv11_SOURCES})
source_group("obs-qsv11\\Header Files" FILES ${obs-qsv11_HEADERS})
source_group("libmfx\\Source Files" FILES ${obs-qsv11_libmfx_SOURCES})
source_group("libmfx\\Header Files" FILES ${obs-qsv11_libmfx_HEADERS})
source_group("text\\Source Files" FILES ${text_SOURCES})
source_group("text\\Header Files" FILES ${text_HEADERS})

install_obs_plugin_with_data(obs-qsv11 data)
