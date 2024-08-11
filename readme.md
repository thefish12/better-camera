# LibTorch
安装libtorch时启用环境变量cxx11 abi
https://github.com/pytorch/pytorch#from-source

# OpenCV
安装opencv启用gstreamer
https://galaktyk.medium.com/how-to-build-opencv-with-gstreamer-b11668fa09c
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules -D WITH_GSTREAMER=ON -D WITH_1394=OFF ../opencv-4.x

dc1394.h:
-D WITH_1394=OFF

呃呃要不还是不用gstreamer了吧，用ffmpeg？
https://medium.com/@vladakuc/compile-opencv-4-7-0-with-ffmpeg-5-compiled-from-the-source-in-ubuntu-434a0bde0ab6

apt install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libgnutls28-dev libmp3lame-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev meson ninja-build pkg-config texinfo wget yasm zlib1g-dev nasm libx264-dev libx265-dev libnuma-dev libvpx-dev libfdk-aac-dev libopus-dev libdav1d-dev

似乎有 libvpx错误。（未解决）
https://blog.csdn.net/weixin_42232238/article/details/106072886
https://blog.csdn.net/quantum7/article/details/104048538

## 先安装docker吧
docker安装完成。
https://docs.docker.com/engine/install/ubuntu/

export http_proxy="http://127.0.0.1:7890"
export https_proxy="https://127.0.0.1:7890"
export ftp_proxy="ftp://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="https://127.0.0.1:7890"
export FTP_PROXY="ftp://127.0.0.1:7890"
