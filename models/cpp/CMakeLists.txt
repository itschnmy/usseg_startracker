cmake_minimum_required(VERSION 3.10)
project(StarTracker)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC does not define M_PI unless this is set
add_definitions(-D_USE_MATH_DEFINES)

# Include directories (Eigen is header-only, bundled locally)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# =====================================================================
# TEST TARGETS (No OpenCV — pure algorithmic verification)
# These only need Eigen (header-only) + our algorithm source files.
# =====================================================================

set(TEST_SOURCES
    estimator.cpp
    function.cpp
    CentroidMatcher.cpp
    StarCatalog.cpp
    PairCatalog.cpp
    VotingIdentifier.cpp
)

# LIS Mode Test (Star Identificator + Attitude Determinator)
add_executable(test_lis ../test/test_lis.cpp ${TEST_SOURCES})

# Tracking Mode Test (CentroidMatcher + Attitude Determinator)
add_executable(test_tracking ../test/test_tracking.cpp ${TEST_SOURCES})

# =====================================================================
# OpenCV Target for Image Preprocessing Verification
# =====================================================================
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Enabling test_image_preprocessing target.")
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(test_image_preprocessing 
        ../test/test_image_preprocessing.cpp 
        ImagePreprocessor/ImagePreprocessor.cpp
    )
    target_include_directories(test_image_preprocessing PRIVATE ImagePreprocessor)
    target_link_libraries(test_image_preprocessing PRIVATE ${OpenCV_LIBS})
else()
    message(STATUS "OpenCV NOT found. Skipping test_image_preprocessing target.")
endif()
