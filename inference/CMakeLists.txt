cmake_minimum_required(VERSION 3.30)
set(VENDDB_INFERENCE_VERSION "0.0.1")
project(venddb_inference VERSION ${VENDDB_INFERENCE_VERSION})

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

FetchContent_Declare(yaml-cpp
	GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
	GIT_PROGRESS TRUE
	GIT_TAG yaml-cpp-0.9.0
	GIT_SHALLOW 1
)
FetchContent_MakeAvailable(yaml-cpp)

# ============================================================================
# Shared library — callable from Go via CGO
# ============================================================================

add_library(venddb_inference SHARED src/lib.cpp)

set_target_properties(venddb_inference PROPERTIES
	POSITION_INDEPENDENT_CODE ON
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS ON
	# Embed the build-directory rpath so the dylib can find its own deps
	BUILD_WITH_INSTALL_RPATH TRUE
	INSTALL_RPATH "@loader_path"
)

target_compile_options(venddb_inference PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-missing-braces
    -Wno-sign-conversion -Wno-deprecated-copy-with-user-provided-copy
    -Wno-deprecated-anon-enum-enum-conversion -Wno-unused-variable)

target_include_directories(venddb_inference
	PUBLIC  include
	PRIVATE ${OpenCV_INCLUDE_DIRS}
	        ${ONNXRUNTIME_DIR}/include/onnxruntime
	        ${yolos_cpp_SOURCE_DIR}/include)

target_link_libraries(venddb_inference PRIVATE ${OpenCV_LIBS} PkgConfig::onnxruntime yaml-cpp::yaml-cpp)

# ============================================================================
# Test/demo executable (links the shared library)
# ============================================================================

add_executable(venddb_inference_demo src/main.cpp)

set_target_properties(venddb_inference_demo PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON
	OUTPUT_NAME "venddb_inference_demo"
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(venddb_inference_demo PRIVATE
	include
	${OpenCV_INCLUDE_DIRS}
	${ONNXRUNTIME_DIR}/include/onnxruntime
	${yolos_cpp_SOURCE_DIR}/include)

target_link_libraries(venddb_inference_demo PRIVATE
	venddb_inference fmt::fmt ${OpenCV_LIBS} PkgConfig::onnxruntime yaml-cpp::yaml-cpp)
