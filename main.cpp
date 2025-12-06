#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp> // Bổ sung để đảm bảo gọi được imread/empty
#include "StarDetector.h"

// Hàm vẽ helper (giữ nguyên logic cũ)
void visualize_results(cv::Mat& img, const std::vector<DetectedStar>& stars) {
    if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    
    for (const auto& s : stars) {
        // Truy cập tọa độ qua Eigen vector
        cv::Point center(std::round(s.position.x()), std::round(s.position.y()));
        
        cv::circle(img, center, (int)(s.radius + 5), cv::Scalar(0, 0, 255), 1);
        cv::drawMarker(img, center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);
    }
}

int main() {
    // 1. CẬP NHẬT: Load ảnh từ file và kiểm tra lỗi
    const std::string filename = "image.png";
    
    // Load ảnh dưới dạng ảnh xám (grayscale)
    cv::Mat raw_image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    
    // KIỂM TRA LỖI BẮT BUỘC
    if (raw_image.empty()) {
        std::cerr << "🔴 LỖI: Khong tim thay file '" << filename 
                  << "' hoac file bi hong. Kiem tra thu muc chay!\n";
        std::cerr << "Ghi chu: File anh phai nam trong thu muc CHUA file StarDetector.exe.\n";
        return -1; // Trả về mã lỗi
    }
    
    // Áp dụng bộ lọc Gaussian nhẹ để làm mịn nhiễu (Thực tế nên làm)
    cv::GaussianBlur(raw_image, raw_image, cv::Size(3, 3), 0);

    // 2. Xử lý
    // Sử dụng threshold sigma 3.5 và diện tích tối thiểu 2 pixel
    StarDetector detector(3.5f, 2); 
    auto stars = detector.process(raw_image);

    // 3. In kết quả (Demo dùng Eigen)
    // Tạo bản sao màu để vẽ lên (vì raw_image là Grayscale)
    cv::Mat display_image = raw_image.clone(); 
    
    std::cout << "\n========== DETECTED STARS ==========\n";
    std::cout << "Total stars detected: " << stars.size() << "\n\n";
    std::cout << std::setw(3) << "ID" << " | "
              << "uBody Vector (3D Unit Vector)\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& s : stars) {
        std::cout << std::setw(3) << s.index << " | ["
                  << std::fixed << std::setprecision(5) 
                  << std::setw(8) << s.uBody(0) << ", "
                  << std::setw(8) << s.uBody(1) << ", "
                  << std::setw(8) << s.uBody(2) << "]\n";
    }
    std::cout << "====================================\n";

    visualize_results(display_image, stars);
    cv::imshow("Detected Stars - Loaded from File", display_image);
    cv::waitKey(0);

    return 0;
}