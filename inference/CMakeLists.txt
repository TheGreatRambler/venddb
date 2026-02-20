cmake_minimum_required(VERSION 3.30)
set(VENDDB_INFERENCE_VERSION "0.0.1")
project(venddb_inference VERSION ${VENDDB_INFERENCE_VERSION})

add_definitions(-DRAPIDJSON_HAS_STDSTRING)

if(CMAKE_BUILD_TYPE EQUAL "DEBUG")
	add_link_options(-g -O0)
endif()

include(ExternalProject)
include(FetchContent)

find_package(PkgConfig REQUIRED)

find_package(OpenCV REQUIRED)

# libfmt for formatting that works
FetchContent_Declare(fmt
	GIT_REPOSITORY https://github.com/fmtlib/fmt.git
	GIT_PROGRESS TRUE
	GIT_TAG 12.1.0
	GIT_SHALLOW 1
)
FetchContent_MakeAvailable(fmt)

find_package(onnxruntime REQUIRED)
pkg_check_modules(onnxruntime REQUIRED IMPORTED_TARGET libonnxruntime)

if(UNIX)
	if(APPLE)
		# This uses homebrew onnxruntime
		# TODO check that user has this
		execute_process(
		    COMMAND brew ruby -e "puts Formulary.factory('onnxruntime').installed_kegs.first.to_s"
		    OUTPUT_VARIABLE ONNXRUNTIME_PREFIX
		    OUTPUT_STRIP_TRAILING_WHITESPACE
		)
		set(ONNXRUNTIME_DIR ${ONNXRUNTIME_PREFIX})
	else()
		# TODO
	endif()
else()
	# TODO
endif()

# Interacting with YOLO26 via ONNX
# This library is header only
FetchContent_Declare(yolos_cpp
	GIT_REPOSITORY https://github.com/Geekgineer/YOLOs-CPP.git
	GIT_PROGRESS TRUE
	GIT_TAG v1.0.0
	GIT_SHALLOW 1
	# Using dumb workaround to just download
	# https://discourse.cmake.org/t/prevent-fetchcontent-makeavailable-to-execute-cmakelists-txt/12704/16
	SOURCE_SUBDIR invalid_dir
)
FetchContent_MakeAvailable(yolos_cpp)

add_executable(venddb_inference ${APPLICATION_TYPE}
	src/main.cpp
)

set_target_properties(venddb_inference PROPERTIES
	POSITION_INDEPENDENT_CODE ON
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS ON
)
target_compile_options(venddb_inference PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-missing-braces
    -Wno-sign-conversion -Wno-deprecated-copy-with-user-provided-copy
    -Wno-deprecated-anon-enum-enum-conversion -Wno-unused-variable)

set_target_properties(venddb_inference PROPERTIES
	OUTPUT_NAME "venddb_inference"
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	CXX_VISIBILITY_PRESET hidden
	POSITION_INDEPENDENT_CODE ON)

target_link_libraries(venddb_inference PRIVATE fmt::fmt ${OpenCV_LIBS} PkgConfig::onnxruntime)
target_include_directories(venddb_inference PRIVATE opencv2 ${ONNXRUNTIME_DIR}/include/onnxruntime ${yolos_cpp_SOURCE_DIR}/include)