#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

// Cấu trúc dữ liệu đầu ra được cập nhật
struct DetectedStar {
    // BỘ ĐỊNH DANH CHÍNH
    int index;                 // ID tạm thời trong frame hiện tại
    Eigen::Vector3d uBody;     // Vector đơn vị 3D trong hệ tọa độ camera (Kết quả mong muốn)
    
    // Dữ liệu hỗ trợ/Trung gian (cần thiết cho Centroiding và Tracking)
    Eigen::Vector2d position;  // Tọa độ 2D Sub-pixel (x, y)
    double intensity;          // Tổng độ sáng (Flux)
    int peak;                  // Giá trị pixel sáng nhất
    double radius;             // Bán kính ước lượng
};

class StarDetector {
public:
    StarDetector(float sigma_threshold = 3.0f, int min_area = 2);
    std::vector<DetectedStar> process(const cv::Mat& image);

private:
    float m_sigma_threshold;
    int m_min_area;
};