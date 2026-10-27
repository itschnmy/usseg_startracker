cmake_minimum_required(VERSION 3.10)
project(StarTracker)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC does not define M_PI unless this is set
if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
endif()

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
add_executable(test_lis ../../examples/tests/test_lis.cpp ${TEST_SOURCES})

# Tracking Mode Test (CentroidMatcher + Attitude Determinator)
add_executable(test_tracking ../../examples/tests/test_tracking.cpp ${TEST_SOURCES})

enable_testing()
add_test(NAME lis COMMAND test_lis)
add_test(NAME tracking COMMAND test_tracking)
set_tests_properties(lis tracking PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../data")

# =====================================================================
# OpenCV Target for Image Preprocessing Verification
# =====================================================================
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Enabling test_image_preprocessing target.")
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(test_image_preprocessing 
        ../../examples/tests/test_image_preprocessing.cpp
        ImagePreprocessor/ImagePreprocessor.cpp
    )
    add_test(NAME image_preprocessing COMMAND test_image_preprocessing
        "${CMAKE_CURRENT_SOURCE_DIR}/../../dataset/calibration/openmv/session_01/03_calibrated.bmp")
    add_executable(star_tracker main.cpp StarDetector.cpp ${TEST_SOURCES})
    target_link_libraries(star_tracker PRIVATE ${OpenCV_LIBS})
    target_include_directories(test_image_preprocessing PRIVATE ImagePreprocessor)
    target_link_libraries(test_image_preprocessing PRIVATE ${OpenCV_LIBS})
else()
    message(STATUS "OpenCV NOT found. Skipping test_image_preprocessing target.")
endif()
