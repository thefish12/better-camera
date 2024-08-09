cmake_minimum_required(VERSION 3.10)

# 设置编译器
# set(CMAKE_C_COMPILER /usr/bin/gcc-11)
# set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
# set(CMAKE_LINKER /usr/bin/ld)

# 项目名称
project(DemoUseTools)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set(Torch_DIR /root/better-camera/libtorch/share/cmake/Torch)
# set(OpenCV_DIR /root/better-camera/opencv/opencv-4.x/cmake)

# 添加源文件
set(SOURCES
    ${CMAKE_SOURCE_DIR}/demo-use-tools.cpp
    ${CMAKE_SOURCE_DIR}/model.cpp
    ${CMAKE_SOURCE_DIR}/tools.cpp
)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR})
# include_directories(/root/better-camera/libtorch/include)
# include_directories(/root/better-camera/opencv/opencv-4.x/include)
# include_directories(/root/better-camera/cpp/opencv/opencv_contrib-4.x/modules/freetype/include)
include_directories(/usr/local/include/)
# include_directories(/root/better-camera/opencv/opencv_contrib-4.x/modules)
# include_directories(/root/better-camera/argparse/include)
# include_directories(/root/ipex/)
# include_directories(/root/ipex/intel-extension-for-pytorch/csrc/cpu/)


# 添加可执行文件
add_executable(demo-use-tools ${SOURCES})

# 链接必要的库
find_package(OpenCV REQUIRED)
# find_package(OpenCV REQUIRED PATHS /root/better-camera/cpp/opencv/build NO_DEFAULT_PATH)
find_package(Torch REQUIRED PATHS /root/better-camera/pytorch/torch/share/cmake/Torch NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Torch REQUIRED)

message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "${OpenCV_INCLUDE_DIRS}")

# add_library(U SHARED tools.cpp model.cpp)

target_link_libraries(demo-use-tools ${OpenCV_LIBS} ${TORCH_LIBRARIES})