#include "ImagePreprocessor.h"
#include <cmath>
#include <algorithm>

ImagePreprocessor::ImagePreprocessor(float bg_scale, float sigma, int filtsize)
    : m_bg_scale(bg_scale), m_sigma(sigma), m_filtsize(filtsize),
      m_crop_enabled(false), m_crop_width(0), m_crop_height(0),
      m_crop_offset_x(0), m_crop_offset_y(0),
      m_crop_offset_x_actual(0), m_crop_offset_y_actual(0) {
    // ESA Tetra3 uses a 3x3 cross structuring element for morphological opening.
    // This helps eliminate single-pixel hot pixels and cosmic ray hits with low computation.
    m_kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
}

void ImagePreprocessor::setCrop(int width, int height, int offset_x, int offset_y) {
    m_crop_enabled = true;
    m_crop_width = width;
    m_crop_height = height;
    m_crop_offset_x = offset_x;
    m_crop_offset_y = offset_y;
}

void ImagePreprocessor::disableCrop() {
    m_crop_enabled = false;
    m_crop_width = 0;
    m_crop_height = 0;
    m_crop_offset_x = 0;
    m_crop_offset_y = 0;
    m_crop_offset_x_actual = 0;
    m_crop_offset_y_actual = 0;
}

bool ImagePreprocessor::preprocess(const cv::Mat& input_image, cv::Mat& clean_output, cv::Mat& binary_output) {
    if (input_image.empty()) {
        return false;
    }

    // 1. Grayscale Conversion (if necessary)
    if (input_image.channels() == 3) {
        cv::cvtColor(input_image, m_gray, cv::COLOR_BGR2GRAY);
    } else {
        m_gray = input_image; // Shallow copy if already grayscale (shares data buffer)
    }

    // 2. Cropping / ROI Selection (if enabled and bounds are valid)
    // Selects a subset of the sensor frame to process. Sub-matrix views in OpenCV
    // share the parent image's memory buffer, guaranteeing zero dynamic allocation.
    cv::Mat processing_img;
    if (m_crop_enabled && m_crop_width > 0 && m_crop_height > 0 && 
        m_crop_width <= m_gray.cols && m_crop_height <= m_gray.rows) {
        
        // Calculate center-relative offsets matching the Tetra3 crop behavior
        int offs_y = (m_gray.rows - m_crop_height) / 2 + m_crop_offset_y;
        int offs_x = (m_gray.cols - m_crop_width) / 2 + m_crop_offset_x;

        // Clamp offsets to be inside the image boundaries
        if (offs_y < 0) offs_y = 0;
        if (offs_y > m_gray.rows - m_crop_height) offs_y = m_gray.rows - m_crop_height;
        if (offs_x < 0) offs_x = 0;
        if (offs_x > m_gray.cols - m_crop_width) offs_x = m_gray.cols - m_crop_width;

        m_crop_offset_x_actual = offs_x;
        m_crop_offset_y_actual = offs_y;

        cv::Rect crop_rect(offs_x, offs_y, m_crop_width, m_crop_height);
        processing_img = m_gray(crop_rect); // Shallow copy view
    } else {
        m_crop_offset_x_actual = 0;
        m_crop_offset_y_actual = 0;
        processing_img = m_gray;
    }

    // 3. Gaussian Blur (3x3 Kernel, Sigma = 1.0)
    // Attenuates high-frequency noise prior to background estimation.
    cv::GaussianBlur(processing_img, m_blur, cv::Size(3, 3), 1.0);

    // 4. TETRA3 DYNAMIC CALIBRATION STAGE: Median Background Estimation
    // Estimating the background map using a local median filter on a downscaled image
    // is highly robust against high-frequency stellar footprints.
    int small_width = cvRound(processing_img.cols * m_bg_scale);
    int small_height = cvRound(processing_img.rows * m_bg_scale);
    
    // Safety guard to ensure the downscaled image is at least min_size x min_size (required for medianBlur)
    int ksize = m_filtsize;
    if (ksize % 2 == 0) ksize += 1; // Median filter size must be odd
    if (ksize < 3) ksize = 3;

    int min_size = std::max(3, ksize);
    if (small_width < min_size) small_width = min_size;
    if (small_height < min_size) small_height = min_size;
    cv::Size small_size(small_width, small_height);

    // Downscale using Nearest-Neighbor interpolation (zero floating point operations)
    cv::resize(m_blur, m_small_bg, small_size, 0, 0, cv::INTER_NEAREST);

    // Median filter to wipe out localized star point footprints
    cv::medianBlur(m_small_bg, m_small_bg, ksize);

    // Upscale back to cropped resolution using Bilinear interpolation to reconstruct smooth background map
    cv::resize(m_small_bg, m_background, processing_img.size(), 0, 0, cv::INTER_LINEAR);

    // Perform pixel-wise clipping subtraction: clean_output = max(blur - background, 0)
    cv::subtract(m_blur, m_background, clean_output);

    // 5. TETRA3 DYNAMIC CALIBRATION STAGE: RMS Noise Estimation & Thresholding
    // Instead of using mean/standard deviation, Tetra3 calculates the standard deviation of noise
    // via a fast global Root Mean Square (RMS) calculation of the subtracted image:
    // RMS = sqrt( sum(I_clean^2) / total_pixels )
    // We compute this efficiently using the L2 Norm: cv::norm(clean_output, cv::NORM_L2)
    double l2_norm = cv::norm(clean_output, cv::NORM_L2);
    double rms_val = l2_norm / std::sqrt(static_cast<double>(clean_output.total()));
    
    // The threshold formula is: Threshold = RMS * sigma
    double threshold_val = m_sigma * rms_val;
    if (threshold_val > 255.0) {
        threshold_val = 255.0;
    }

    // Binarize the background-subtracted image
    cv::threshold(clean_output, binary_output, threshold_val, 255, cv::THRESH_BINARY);

    // 6. Morphological Opening
    // Applies the 3x3 cross kernel to eradicate isolated cosmic ray triggers and hot pixels
    cv::morphologyEx(binary_output, binary_output, cv::MORPH_OPEN, m_kernel);

    return true;
}


