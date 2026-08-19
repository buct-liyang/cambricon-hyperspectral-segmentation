cmake_minimum_required(VERSION 3.5)
project(cambricon_hyperspectral_segmentation)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-Wl,--copy-dt-needed-entries -g")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ---[ neuware
find_package(Neuware)
include_directories(${NEUWARE_INCLUDE_DIR})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/neuware/lib64)
list(APPEND 3RDPARTY_LIBS ${CNRT_LIBS} ${CNCODEC_LIBS})
# ---[ easydk

include_directories(3rdparty/gflags/include)
link_directories(3rdparty/gflags/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glog/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glog/lib)
# ---[ easydk

# ---[ opencv
#set(OpenCV_FIND_QUIETLY true)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv)
#find_package(OpenCV REQUIRED) //this will include opencv_ts which result in crash on centos
find_package(OpenCV OPTIONAL_COMPONENTS imgcodecs CONFIG PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH)
find_package(OpenCV OPTIONAL_COMPONENTS videoio CONFIG PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH)
set(optional_opencv_libs ${OpenCV_LIBS})
find_package(OpenCV REQUIRED core imgproc highgui features2d CONFIG PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH)
if(OpenCV_FOUND)
    if(imgcodecs_FOUND)
        list(APPEND OpenCV_LIBS optional_opencv_libs)
    endif()
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND 3RDPARTY_LIBS ${OpenCV_LIBS})
    message(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "opencv libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "opencv not found!")
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/common common_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/common/postprocess postprocess_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/common/preprocess preprocess_srcs)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link=${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/neuware/lib64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link=${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ffmpeg/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link=${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/x264/lib")

list(APPEND srcs ${common_srcs} ${postprocess_srcs} ${preprocess_srcs})

# 设置需要排除的文件名的正则表达式
#set(EXCLUDE_PATTERN "${CMAKE_CURRENT_SOURCE_DIR}/common/WHY.*\\.cpp")
#list(FILTER srcs EXCLUDE REGEX ${EXCLUDE_PATTERN})
#message(STATUS "srcs is ${srcs}")

#[[
#AI算法动态链接库-ssd
add_library(ssd SHARED ${CMAKE_CURRENT_SOURCE_DIR}/common/ssd.cpp ${srcs})
target_link_libraries(ssd PRIVATE easydk)
target_compile_definitions(ssd PRIVATE HAVE_CNCV)
target_include_directories(ssd PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include/infer_server/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common)
]]

#AI算法动态链接库-Seg
add_library(Seg SHARED  ${CMAKE_CURRENT_SOURCE_DIR}/samples/seg.cpp ${srcs})
        #${CMAKE_CURRENT_SOURCE_DIR}/common/postprocess/seg_postproc.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/common/preprocess/seg_preproc.cpp)
target_link_libraries(Seg PRIVATE easydk)
target_compile_definitions(Seg PRIVATE HAVE_CNCV)
target_include_directories(Seg PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include/infer_server/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

#AI算法动态链接库-Seg(方案v2：整图固定归一化)
add_library(SegFixedNorm SHARED  ${CMAKE_CURRENT_SOURCE_DIR}/samples/seg_fixed_norm.cpp ${srcs})
target_link_libraries(SegFixedNorm PRIVATE easydk)
target_compile_definitions(SegFixedNorm PRIVATE HAVE_CNCV)
target_include_directories(SegFixedNorm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include/infer_server/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

#[[
#AI算法动态链接库-resnet18_plane
add_library(plane SHARED  ${CMAKE_CURRENT_SOURCE_DIR}/common/resnet18_plane.cpp ${srcs}
target_link_libraries(plane PRIVATE easydk)
target_compile_definitions(plane PRIVATE HAVE_CNCV)
target_include_directories(plane PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include/infer_server/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common)
]]
#[[
#AI算法动态链接库-resnet18_ship
add_library(ship SHARED  ${CMAKE_CURRENT_SOURCE_DIR}/common/resnet18_ship.cpp ${srcs})
target_link_libraries(ship PRIVATE easydk)
target_compile_definitions(ship PRIVATE HAVE_CNCV)
target_include_directories(ship PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include/infer_server/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common)
]]

#自测试Demo

add_executable(infer_demo samples/unittest_seg.cpp common/cnosd.cpp common/dataframe.cpp common/collection.cpp common/infer_object.cpp)
target_link_libraries(infer_demo PRIVATE easydk pthread dl ${OpenCV_LIBS})
target_include_directories(infer_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easydk/include/infer_server)
