SET(pwd ${CMAKE_CURRENT_LIST_DIR})

UNSET(src)


# add_definitions(-DCOMMAND_LINE_ONLY)
# Uncomment the following lines to explicitly set files to compile from
# CMakeLists.sources
#IF (EXISTS CMakeLists.sources)
#      FILE(STRINGS CMakeLists.sources src)
#ENDIF()
 
# By default, source files are added automatically
IF (NOT src)
	AUX_SOURCE_DIRECTORY(${pwd} src)
	AUX_SOURCE_DIRECTORY(${pwd}/fileio src)
	AUX_SOURCE_DIRECTORY(${pwd}/parser src)
	AUX_SOURCE_DIRECTORY(${pwd}/scene src)
	AUX_SOURCE_DIRECTORY(${pwd}/SceneObjects src)
	AUX_SOURCE_DIRECTORY(${pwd}/ui src)
	IF (WIN32)
		AUX_SOURCE_DIRECTORY(${pwd}/win32 src)
	ENDIF (WIN32)
ENDIF(NOT src)
add_executable(ray ${src})

message(STATUS "ray added, files ${src}")

target_link_libraries(ray ${OPENGL_gl_LIBRARY})


# Manual FLTK
# target_link_libraries(main ${CMAKE_SOURCE_DIR}/libbingitup.a)
include_directories(${CMAKE_SOURCE_DIR}/fltk-1.3.8/)
link_directories(${CMAKE_SOURCE_DIR}/fltk-1.3.8/lib)
target_link_libraries(ray ${CMAKE_SOURCE_DIR}/fltk-1.3.8/lib/libfltk.a)
target_link_libraries(ray ${CMAKE_SOURCE_DIR}/fltk-1.3.8/lib/libfltk_gl.a)
target_link_libraries(ray ${CMAKE_SOURCE_DIR}/fltk-1.3.8/lib/libfltk_images.a)
target_link_libraries(ray ${CMAKE_SOURCE_DIR}/fltk-1.3.8/lib/libfltk_forms.a)
target_link_libraries(ray X11 dl Xext Xfixes Xrender Xcursor Xinerama)

# SET(FLTK_SKIP_FLUID TRUE)
# FIND_PACKAGE(FLTK REQUIRED)
# SET_PROPERTY(TARGET ray APPEND PROPERTY INCLUDE_DIRECTORIES ${FLTK_INCLUDE_DIRS})
# SET_PROPERTY(TARGET ray APPEND PROPERTY INCLUDE_DIRECTORIES ${FLTK_INCLUDE_DIR})
# if(WIN32)
# 	set(FLTK_LIBRARIES fltk;fltk_gl)
# endif()
# target_link_libraries(ray ${FLTK_LIBRARIES})
FIND_PACKAGE(JPEG REQUIRED)
target_link_libraries(ray ${JPEG_LIBRARIES})
FIND_PACKAGE(PNG REQUIRED)
target_link_libraries(ray ${PNG_LIBRARIES})
FIND_PACKAGE(ZLIB REQUIRED)
target_link_libraries(ray ${ZLIB_LIBRARIES})
SET_PROPERTY(TARGET ray APPEND PROPERTY INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIR})
target_link_libraries(ray ${OPENGL_glu_LIBRARY})
