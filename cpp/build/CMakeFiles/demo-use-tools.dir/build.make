# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /root/anaconda3/lib/python3.12/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /root/anaconda3/lib/python3.12/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/better-camera/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/better-camera/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/demo-use-tools.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/demo-use-tools.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/demo-use-tools.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo-use-tools.dir/flags.make

CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o: CMakeFiles/demo-use-tools.dir/flags.make
CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o: /root/better-camera/cpp/demo-use-tools.cpp
CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o: CMakeFiles/demo-use-tools.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/root/better-camera/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o -MF CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o.d -o CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o -c /root/better-camera/cpp/demo-use-tools.cpp

CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.i"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/better-camera/cpp/demo-use-tools.cpp > CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.i

CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.s"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/better-camera/cpp/demo-use-tools.cpp -o CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.s

CMakeFiles/demo-use-tools.dir/model.cpp.o: CMakeFiles/demo-use-tools.dir/flags.make
CMakeFiles/demo-use-tools.dir/model.cpp.o: /root/better-camera/cpp/model.cpp
CMakeFiles/demo-use-tools.dir/model.cpp.o: CMakeFiles/demo-use-tools.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/root/better-camera/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/demo-use-tools.dir/model.cpp.o"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo-use-tools.dir/model.cpp.o -MF CMakeFiles/demo-use-tools.dir/model.cpp.o.d -o CMakeFiles/demo-use-tools.dir/model.cpp.o -c /root/better-camera/cpp/model.cpp

CMakeFiles/demo-use-tools.dir/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/demo-use-tools.dir/model.cpp.i"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/better-camera/cpp/model.cpp > CMakeFiles/demo-use-tools.dir/model.cpp.i

CMakeFiles/demo-use-tools.dir/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/demo-use-tools.dir/model.cpp.s"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/better-camera/cpp/model.cpp -o CMakeFiles/demo-use-tools.dir/model.cpp.s

CMakeFiles/demo-use-tools.dir/tools.cpp.o: CMakeFiles/demo-use-tools.dir/flags.make
CMakeFiles/demo-use-tools.dir/tools.cpp.o: /root/better-camera/cpp/tools.cpp
CMakeFiles/demo-use-tools.dir/tools.cpp.o: CMakeFiles/demo-use-tools.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/root/better-camera/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/demo-use-tools.dir/tools.cpp.o"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo-use-tools.dir/tools.cpp.o -MF CMakeFiles/demo-use-tools.dir/tools.cpp.o.d -o CMakeFiles/demo-use-tools.dir/tools.cpp.o -c /root/better-camera/cpp/tools.cpp

CMakeFiles/demo-use-tools.dir/tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/demo-use-tools.dir/tools.cpp.i"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/better-camera/cpp/tools.cpp > CMakeFiles/demo-use-tools.dir/tools.cpp.i

CMakeFiles/demo-use-tools.dir/tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/demo-use-tools.dir/tools.cpp.s"
	/root/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/better-camera/cpp/tools.cpp -o CMakeFiles/demo-use-tools.dir/tools.cpp.s

# Object files for target demo-use-tools
demo__use__tools_OBJECTS = \
"CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o" \
"CMakeFiles/demo-use-tools.dir/model.cpp.o" \
"CMakeFiles/demo-use-tools.dir/tools.cpp.o"

# External object files for target demo-use-tools
demo__use__tools_EXTERNAL_OBJECTS =

demo-use-tools: CMakeFiles/demo-use-tools.dir/demo-use-tools.cpp.o
demo-use-tools: CMakeFiles/demo-use-tools.dir/model.cpp.o
demo-use-tools: CMakeFiles/demo-use-tools.dir/tools.cpp.o
demo-use-tools: CMakeFiles/demo-use-tools.dir/build.make
demo-use-tools: /usr/local/lib/libopencv_gapi.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_stitching.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_aruco.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_bgsegm.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_bioinspired.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_ccalib.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_dnn_objdetect.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_dnn_superres.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_dpm.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_face.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_freetype.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_fuzzy.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_hdf.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_hfs.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_img_hash.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_intensity_transform.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_line_descriptor.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_mcc.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_quality.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_rapid.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_reg.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_rgbd.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_saliency.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_signal.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_stereo.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_structured_light.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_superres.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_surface_matching.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_tracking.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_videostab.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_wechat_qrcode.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_xfeatures2d.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_xobjdetect.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_xphoto.so.4.10.0
demo-use-tools: /root/better-camera/pytorch/torch/lib/libtorch.so
demo-use-tools: /root/better-camera/pytorch/torch/lib/libc10.so
demo-use-tools: /root/better-camera/pytorch/torch/lib/libkineto.a
demo-use-tools: /usr/local/lib/libopencv_shape.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_highgui.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_datasets.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_plot.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_text.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_ml.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_phase_unwrapping.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_optflow.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_ximgproc.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_video.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_videoio.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_imgcodecs.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_objdetect.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_calib3d.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_dnn.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_features2d.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_flann.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_photo.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_imgproc.so.4.10.0
demo-use-tools: /usr/local/lib/libopencv_core.so.4.10.0
demo-use-tools: /root/better-camera/pytorch/torch/lib/libc10.so
demo-use-tools: CMakeFiles/demo-use-tools.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/root/better-camera/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable demo-use-tools"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo-use-tools.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo-use-tools.dir/build: demo-use-tools
.PHONY : CMakeFiles/demo-use-tools.dir/build

CMakeFiles/demo-use-tools.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo-use-tools.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo-use-tools.dir/clean

CMakeFiles/demo-use-tools.dir/depend:
	cd /root/better-camera/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/better-camera/cpp /root/better-camera/cpp /root/better-camera/cpp/build /root/better-camera/cpp/build /root/better-camera/cpp/build/CMakeFiles/demo-use-tools.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/demo-use-tools.dir/depend
