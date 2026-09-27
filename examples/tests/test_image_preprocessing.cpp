#include "ImagePreprocessor.h"
#include <iostream>
#include <string>
#include <cmath>

#ifdef _WIN32
#include <direct.h> // for _mkdir
#else
#include <sys/stat.h> // for mkdir
#endif

// Helper to create directory structure sequentially
void createDirectory(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "  TEST: IMAGE PREPROCESSOR (ESA Tetra3 Model)\n";
    std::cout << "==================================================\n\n";

    // 1. Ensure output directory exists
    createDirectory("test");
    createDirectory("test/results");
    createDirectory("test/results/image_preprocessing");

    // 2. Define image paths
    std::string input_path = "test/img/9cee97e5-44a4-4193-9384-586386b7ab85.bmp";
    std::string output_clean_path = "test/results/image_preprocessing/clean.png";
    std::string output_binary_path = "test/results/image_preprocessing/binary.png";
    std::string output_roi_path = "test/results/image_preprocessing/cropped_roi.png";

    std::cout << "Loading input image: " << input_path << " ...\n";
    cv::Mat input_img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);

    if (input_img.empty()) {
        std::cerr << "FAIL: Cannot load test image at " << input_path << "\n";
        std::cerr << "Please verify the image exists in test/img/\n";
        return 1;
    }

    std::cout << "Original image size: " << input_img.cols << " x " << input_img.rows << "\n\n";

    // 3. Initialize Preprocessor
    // Using default Tetra3 parameters: bg_scale = 0.03125 (1/32), sigma = 3.0, filtsize = 3 (median filter)
    float bg_scale = 0.03125f;
    float sigma = 3.0f;
    int filtsize = 3;
    ImagePreprocessor preprocessor(bg_scale, sigma, filtsize);

    // 4. Configure Cropping / ROI Selection (Kowa LM16HC lens distortion mitigation)
    // We select a 1024x1024 region near the center, offset by (dx=50, dy=-30) to verify offsets
    int crop_w = 1024;
    int crop_h = 1024;
    int offset_x = 50;
    int offset_y = -30;
    preprocessor.setCrop(crop_w, crop_h, offset_x, offset_y);

    std::cout << "Configuring Cropping (ROI Selection):\n";
    std::cout << "  Requested Size   : " << crop_w << " x " << crop_h << "\n";
    std::cout << "  Requested Offset : (" << offset_x << ", " << offset_y << ")\n";

    // 5. Run Preprocessor
    cv::Mat clean_img, binary_img;
    
    // Warm-up run (allocates cached buffers)
    bool ok = preprocessor.preprocess(input_img, clean_img, binary_img);
    if (!ok) {
        std::cerr << "FAIL: Preprocessing failed!\n";
        return 1;
    }

    // Benchmark run to verify zero-allocation speed
    double t0 = static_cast<double>(cv::getTickCount());
    preprocessor.preprocess(input_img, clean_img, binary_img);
    double t_elapsed = (static_cast<double>(cv::getTickCount()) - t0) / cv::getTickFrequency() * 1000.0;

    cv::Point actual_offset = preprocessor.getCropOffset();
    std::cout << "Preprocessing Succeeded:\n";
    std::cout << "  Actual Crop Offset : (" << actual_offset.x << ", " << actual_offset.y << ")\n";
    std::cout << "  Processed ROI Size : " << clean_img.cols << " x " << clean_img.rows << "\n";
    std::cout << "  Execution Time     : " << t_elapsed << " ms\n\n";

    // 6. Save results
    std::cout << "Saving clean (background subtracted) image to: " << output_clean_path << " ...\n";
    cv::imwrite(output_clean_path, clean_img);

    std::cout << "Saving binary (thresholded & opened) mask to: " << output_binary_path << " ...\n";
    cv::imwrite(output_binary_path, binary_img);

    // Create a visualization showing where the cropped ROI sits in the original frame
    cv::Mat roi_vis;
    cv::cvtColor(input_img, roi_vis, cv::COLOR_GRAY2BGR);
    cv::Rect crop_rect(actual_offset.x, actual_offset.y, clean_img.cols, clean_img.rows);
    cv::rectangle(roi_vis, crop_rect, cv::Scalar(0, 255, 0), 4); // Green bounding box
    
    std::cout << "Saving cropped ROI visualization to: " << output_roi_path << " ...\n";
    cv::imwrite(output_roi_path, roi_vis);

    std::cout << "\n>>> Preprocessor test: SUCCESS! <<<\n";
    return 0;
}
