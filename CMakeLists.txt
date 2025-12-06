cmake_minimum_required(VERSION 3.10)
project(StarDetectorEigen)

# 1. Tìm OpenCV
find_package(OpenCV REQUIRED)

# 2. Định nghĩa đường dẫn tới thư mục gốc của Eigen
set(EIGEN_ROOT_DIR "${CMAKE_SOURCE_DIR}")

# 3. KHAI BÁO MỤC TIÊU (ADD EXECUTABLE) NGAY TẠI ĐÂY!
# CMake phải biết "StarDetector" là gì trước khi cấu hình nó.
add_executable(StarDetector 
    main.cpp 
    StarDetector.cpp
)

# 4. ÁP DỤNG CẤU HÌNH CHO MỤC TIÊU (Bây giờ đã hợp lệ)
target_include_directories(StarDetector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN_ROOT_DIR}
)

# 5. Link thư viện (Cũng phải sau add_executable)
target_link_libraries(StarDetector 
    ${OpenCV_LIBS}
)