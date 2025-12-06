#include "StarDetector.h"
#include <cmath>

// Hàm GIẢ LẬP: Tính Vector Đơn vị 3D (uBody)
// CHÚ Ý: Trong hệ thống thực tế, hàm này cần sử dụng các tham số hiệu chuẩn camera.
Eigen::Vector3d calculate_uBody(double x, double y) {
    // Giả định các tham số nội tại (Intrinsic Parameters)
    double cx = 320.0; // Tâm ảnh X (Giả định)
    double cy = 240.0; // Tâm ảnh Y (Giả định)
    double f = 500.0;  // Tiêu cự (Focal length) (Giả định)

    // Chuyển đổi tọa độ pixel (x, y) sang tọa độ chuẩn hóa (normalized coordinates)
    double x_norm = (x - cx) / f;
    double y_norm = (y - cy) / f;
    
    // Tạo vector 3D (X/Z, Y/Z, 1) và chuẩn hóa
    Eigen::Vector3d u(x_norm, y_norm, 1.0);
    
    return u.normalized(); // Trả về vector đơn vị
}

StarDetector::StarDetector(float sigma_threshold, int min_area)
    : m_sigma_threshold(sigma_threshold), m_min_area(min_area) {}

std::vector<DetectedStar> StarDetector::process(const cv::Mat& image) {
    // 1. Thresholding và tìm Contours (Giữ nguyên logic)
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    cv::Scalar bg_mean, bg_std;
    cv::meanStdDev(gray, bg_mean, bg_std);
    double threshold_val = bg_mean[0] + (m_sigma_threshold * bg_std[0]);
    if (threshold_val > 255) threshold_val = 255;

    cv::Mat binary;
    cv::threshold(gray, binary, threshold_val, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<DetectedStar> detected_stars;
    int star_id = 0;

    // 2. Tính Centroid và tạo DetectedStar
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < m_min_area) continue;

        cv::Rect box = cv::boundingRect(contour);
        cv::Mat roi = gray(box);

        // Tính Centroid 2D
        cv::Mat roi_float;
        roi.convertTo(roi_float, CV_32F);
        cv::Scalar bg_mean_roi = cv::mean(roi_float);
        cv::Mat roi_subtracted = roi_float - bg_mean_roi[0];
        cv::max(roi_subtracted, 0.0, roi_subtracted);
        cv::Moments M = cv::moments(roi_subtracted);
        if (M.m00 == 0) continue;

        double cx_local = M.m10 / M.m00;
        double cy_local = M.m01 / M.m00;
        double global_x = box.x + cx_local;
        double global_y = box.y + cy_local;

        double min_v, max_v;
        cv::minMaxLoc(roi, &min_v, &max_v);

        // TẠO OBJECT KẾT QUẢ
        DetectedStar star;
        
        // Gán index (bộ định danh 1)
        star.index = star_id++;
        
        // Gán dữ liệu trung gian
        star.position = Eigen::Vector2d(global_x, global_y);
        star.intensity = M.m00;
        star.peak = static_cast<int>(max_v);
        star.radius = std::sqrt(area / CV_PI);

        // TÍNH TOÁN uBody (bộ định danh 2)
        star.uBody = calculate_uBody(global_x, global_y);

        detected_stars.push_back(star);
    }

    return detected_stars;
}