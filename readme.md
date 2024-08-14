# LibTorch
安装libtorch时启用环境变量cxx11 abi
https://github.com/pytorch/pytorch#from-source

# OpenCV
安装opencv使用ffmpeg
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules -D WITH_GSTREAMER=OFF -D WITH_FFMPEG=ON -D WITH_1394=OFF ../opencv-4.x

安装ffmpeg依赖libavcodec libavformat libavutil libswscale

dc1394.h:
-D WITH_1394=OFF


## 先安装docker吧
docker安装完成。
https://docs.docker.com/engine/install/ubuntu/

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export ftp_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export FTP_PROXY=http://127.0.0.1:7890
