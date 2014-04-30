cmake_minimum_required(VERSION 2.4.6)

project(but_calibration_evolution)
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -g")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(include)

find_package(PCL 1.7 REQUIRED) 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS}) 
add_definitions(${PCL_DEFINITIONS}) 

add_executable(calibration src/calibration.cpp src/Image.cpp src/Velodyne.cpp src/Similarity.cpp ) 
target_link_libraries(calibration ${PCL_LIBRARIES}) 
target_link_libraries(calibration ${OpenCV_LIBS})

