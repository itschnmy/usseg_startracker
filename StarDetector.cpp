#include "StarDetector.h"
#include <cmath>

StarDetector::StarDetector(float sigma_threshold, int min_area)
    : m_sigma_threshold(sigma_threshold), m_min_area(min_area) {}

std::vector<DetectedStar> StarDetector::process(const cv::Mat& image) {
    // 1. Thresholding và tìm Contours
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

        // Đặt mặc định uBody bằng vector 0 (sẽ được cập nhật bằng CameraModel bên ngoài)
        star.uBody = Eigen::Vector3d::Zero();

        detected_stars.push_back(star);
    }

    return detected_stars;
}
