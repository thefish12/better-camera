cmake_minimum_required(VERSION 2.8)
project( example-app )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# include_directories(/root/better-camera/cpp/opencv/opencv_contrib-4.x/modules/freetype/src)
# include_directories(/root/better-camera/cpp/opencv/opencv_contrib-4.x/modules/freetype/include)

add_executable( example-app example-app.cpp )
target_link_libraries( example-app ${OpenCV_LIBS} )