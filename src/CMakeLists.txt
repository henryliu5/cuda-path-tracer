SET(pwd ${CMAKE_CURRENT_LIST_DIR})

UNSET(src)


# add_definitions(-DCOMMAND_LINE_ONLY)
# Uncomment the following lines to explicitly set files to compile from
# CMakeLists.sources
IF (EXISTS ${CMAKE_SOURCE_DIR}/src/CMakeLists.sources)
     FILE(STRINGS CMakeLists.sources custom_src)
ENDIF()
 

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

# Build with CUDA
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
find_package(CUDA 9.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_libraries(${CUDA_LIBRARY_DIRS})

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -c" )

add_executable(ray ${src} ${custom_src})
# target_compile_features(ray PUBLIC cxx_std_14)
set_target_properties(ray PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
# 	get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
# 	if(NOT "${old_flags}" STREQUAL "")
# 		string(REPLACE ";" "," CUDA_flags "${old_flags}")
# 		set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
# 			"$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
# 			)
# 	endif()
# endfunction()
# CUDA_CONVERT_FLAGS(ray)


message(STATUS "ray added, files ${src} ${custom_src}")

target_link_libraries(ray ${OPENGL_gl_LIBRARY})

SET(FLTK_SKIP_FLUID TRUE)
FIND_PACKAGE(FLTK REQUIRED)
SET_PROPERTY(TARGET ray APPEND PROPERTY INCLUDE_DIRECTORIES ${FLTK_INCLUDE_DIRS})
SET_PROPERTY(TARGET ray APPEND PROPERTY INCLUDE_DIRECTORIES ${FLTK_INCLUDE_DIR})
if(WIN32)
	set(FLTK_LIBRARIES fltk;fltk_gl)
endif()
target_link_libraries(ray ${FLTK_LIBRARIES})
FIND_PACKAGE(JPEG REQUIRED)
target_link_libraries(ray ${JPEG_LIBRARIES})
FIND_PACKAGE(PNG REQUIRED)
target_link_libraries(ray ${PNG_LIBRARIES})
FIND_PACKAGE(ZLIB REQUIRED)
target_link_libraries(ray ${ZLIB_LIBRARIES})
SET_PROPERTY(TARGET ray APPEND PROPERTY INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIR})
target_link_libraries(ray ${OPENGL_glu_LIBRARY})


# device info executable
add_executable(deviceInfo ./deviceInfo.cu)
target_compile_options(deviceInfo PRIVATE -w) #suppress warnings
set_target_properties(deviceInfo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)