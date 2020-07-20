cmake_minimum_required(VERSION 3.1)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(glfw3 REQUIRED)

include_directories(src)

IF(APPLE)
   INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
   FIND_LIBRARY(COCOA_LIBRARY Cocoa)
   FIND_LIBRARY(OpenGL_LIBRARY OpenGL )
   MARK_AS_ADVANCED (COCOA_LIBRARY
                     GLUT_LIBRARY
                     OpenGL_LIBRARY)
   SET(EXTRA_LIBS ${COCOA_LIBRARY} ${GLUT_LIBRARY} ${OpenGL_LIBRARY})
ENDIF (APPLE)

set (DEBUG_FLAG "-g")
add_definitions(${DEBUG_FLAG})
add_executable(main src/crowds_cpu_orginal.cpp)
#target_link_libraries(main glfw ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_libraries(main glfw ${OPENGL_LIBRARIES}  )
