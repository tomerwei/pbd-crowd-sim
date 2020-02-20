cmake_minimum_required(VERSION 3.1)

IF(APPLE)
   INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
   FIND_LIBRARY(COCOA_LIBRARY Cocoa)
   FIND_LIBRARY(GLUT_LIBRARY GLUT )
   FIND_LIBRARY(OpenGL_LIBRARY OpenGL )
   MARK_AS_ADVANCED (COCOA_LIBRARY
                     GLUT_LIBRARY
                     OpenGL_LIBRARY)
   SET(EXTRA_LIBS ${COCOA_LIBRARY} ${GLUT_LIBRARY} ${OpenGL_LIBRARY})
ENDIF (APPLE)

find_package(CUDA QUIET REQUIRED)

include_directories(/opt/local/include
            /usr/local/include
            )
set (DYN_LINK       /opt/local/lib
                    /usr/local/lib)
#set (MORE_INFO_FLAG "-g -lineinfo  -gencode arch=compute_30,code=sm_30 ")
#add_definitions(${MORE_INFO_FLAG})

set (DEBUG_FLAG "-g")
add_definitions(${DEBUG_FLAG})

link_directories(${DYN_LINK})
find_package(glfw3 REQUIRED)
include_directories(src)
include_directories(/opt/local/include)
#include_directories(/Users/tomerwei/NVIDIA_CUDA-8.0_Samples/common/inc)
link_directories(/opt/local/lib)


#target_link_libraries(main ${EXTRA_LIBS})
# Specify binary name and source file to build it from
cuda_add_executable(
    main
    src/crowds_cpu_orginal.cu)
    
target_link_libraries(main glfw ${EXTRA_LIBS})

