set(CAFFE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CAFFE_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_definitions(-DUSE_EIGEN)

if(PROTOBUF_ROOT)
	INCLUDE_DIRECTORIES(${PROTOBUF_ROOT}/include)
endif()

if(NOT MSVC)
	SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++0x -Wno-sign-compare")
	SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++0x -Wno-sign-compare" )
endif()

add_subdirectory(src/caffe)
