cmake_minimum_required(VERSION 3.2) 
project(miniFEM)

option(MINI_QUIET_DOWNLOAD "Give minimal output when downloading optional libraries" ON)
option(MINI_ENABLE_TESTS "Build google tests" ON)
option(MINI_ENABLE_BENCHMARKS "Build google benchmarks" ON)

# The version number. It doesn't do anything yet
set (VERSION_MAJOR 0)
set (VERSION_MINOR 1)     

FIND_PACKAGE( OpenMP )
if(OPENMP_FOUND)
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#TODO: these directories and definitions don't make much sense. Fix
if(UNIX)
	include_directories(/opt/local/include/eigen3)
	add_definitions(-std=c++11 -stdlib=libc++ -DNDEBUG -DEIGEN_NO_DEBUG -O3 -g -Wall -DMACOS)
else()
	find_path(EIGEN_INCLUDE_DIR Eigen3)
	if(EIGEN_INCLUDE_DIR STREQUAL "EIGEN_INCLUDE_DIR-NOTFOUND")
	  message(FATAL_ERROR "Could not find Eigen Library.")
	endif()
	include_directories(${EIGEN_INCLUDE_DIR})
endif()

set(HDRS ${CMAKE_CURRENT_SOURCE_DIR}/minifem.h)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
 
add_library(${PROJECT_NAME} INTERFACE) 
target_sources(${PROJECT_NAME} INTERFACE ${HDRS})
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}/include>
  )
 
# tell CMake where to find *.cmake module files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
	${PROJECT_SOURCE_DIR}/cmake
    )

if(MINI_QUIET_DOWNLOAD)
	set(QUIET_DOWNLOAD_VAR QUIET)
else()
	set(QUIET_DOWNLOAD_VAR "")
endif()
	
if(MINI_ENABLE_TESTS)
	find_package(GTest QUIET)
	if(${GTEST_FOUND})
		message(STATUS "Using installed version of GTEST.")
		#NOTE: I think this should work but I didn't test it yet.
		include_directories(${GTEST_INCLUDE_DIRS})
	else()
		include(DownloadProject)
		download_project(PROJ                googletest
						 GIT_REPOSITORY      https://github.com/google/googletest.git
						 GIT_TAG             master
						 ${QUIET_DOWNLOAD_VAR}
		)
		# Prevent overriding the parent project's compiler/linker
		# settings on Windows
		set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
		# Add googletest directly to our build.
		add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
						 ${CMAKE_BINARY_DIR}/googletest-build)
	endif()
	#group gtest targets in a folder to avoid crowding the IDE
	set_target_properties(gtest PROPERTIES FOLDER thirdparty)
	set_target_properties(gtest_main PROPERTIES FOLDER thirdparty)
	set_target_properties(gmock PROPERTIES FOLDER thirdparty)
	set_target_properties(gmock_main PROPERTIES FOLDER thirdparty)
	#add the tests
	file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
	add_executable(tests ${TEST_SOURCES})
	target_link_libraries(tests gtest_main ${PROJECT_NAME})
	#include(CTest)
	enable_testing()
	add_test(NAME dummy_test COMMAND tests)
endif()

if(MINI_ENABLE_BENCHMARKS)
	include(DownloadProject)
	download_project(PROJ                googlebenchmark
					 GIT_REPOSITORY      https://github.com/google/benchmark.git
					 GIT_TAG             master
					 ${QUIET_DOWNLOAD_VAR}
	)
	#disabling the tests on google benchmark and creating the benchmark target.
	SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
	message("Output from adding google benchmark:")
	add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
	#Put the benchark target in a folder to avoid crowding the IDE
	set_target_properties(benchmark PROPERTIES FOLDER thirdparty)
	#Create the benchmarks
	file(GLOB_RECURSE ALL_BENCH_CPP ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp)
	file(GLOB_RECURSE BENCHHDRS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.h)
	foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})
		get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)
		# Avoid name collision 
		set(TARGET_NAME Bench_${ONE_BENCH_EXEC})
		add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
		set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
    target_link_libraries(${TARGET_NAME} ${PROJECT_NAME} benchmark
                          ${CMAKE_THREAD_LIBS_INIT})
		# If you want to run benchmarks with the "make test" command, uncomment me
		add_test(${TARGET_NAME} ${ONE_BENCH_EXEC})
		#Grouping all the benchmarks in a single folder
		set_target_properties(${TARGET_NAME} PROPERTIES FOLDER benchmarks)
		target_include_directories(${TARGET_NAME} PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/benchmarks>)
		target_sources(${TARGET_NAME} PRIVATE ${BENCHHDRS})
	endforeach()
	#Copy the input files used in some benchmarks
	file(GLOB INPUTBENCH ./benchmarks/input/*.inp)
	file(COPY ${INPUTBENCH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/benchmark/input)
endif()








 











