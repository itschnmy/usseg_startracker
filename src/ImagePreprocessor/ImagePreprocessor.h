#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Class executing the advanced image preprocessing pipeline for Star Tracker.
 * Aligns with the European Space Agency's (ESA) Tetra3 dynamic calibration pipeline.
 * Optimizes memory consumption by caching internal buffers to prevent dynamic allocation.
 */
class ImagePreprocessor {
private:
    // Configurable parameters
    float m_bg_scale;        // Scale factor for background map estimation (e.g. 1/32 or 1/16)
    float m_sigma;           // Threshold multiplier (sigma)
    int m_filtsize;          // Filter size for local median background subtraction (median filter)
    
    // Cropping (ROI) parameters
    bool m_crop_enabled;
    int m_crop_width;
    int m_crop_height;
    int m_crop_offset_x;     // Target offset right from center
    int m_crop_offset_y;     // Target offset down from center
    
    // Actual calculated crop offsets (used to map centroids back to original image)
    int m_crop_offset_x_actual;
    int m_crop_offset_y_actual;

    // Cached buffers to avoid dynamic allocation during runtime loop
    cv::Mat m_gray;
    cv::Mat m_blur;
    cv::Mat m_small_bg;
    cv::Mat m_background;
    cv::Mat m_kernel;

public:
    /**
     * @brief Construct a new Image Preprocessor object
     * @param bg_scale Scaling factor for downscaling-upscaling background estimation
     * @param sigma Standard deviation (noise) multiplier for thresholding
     * @param filtsize Size of local median filter (must be odd, default 3)
     */
    ImagePreprocessor(float bg_scale = 0.03125f, float sigma = 3.0f, int filtsize = 3);

    /**
     * @brief Executes the preprocessing pipeline on an input image.
     * 
     * @param input_image Original sensor image (Grayscale or BGR).
     * @param clean_output Output matrix to store the background-subtracted image (I_clean).
     * @param binary_output Output matrix to store the binarized and morphologically opened mask (binary).
     * @return true if preprocessing succeeded, false otherwise.
     */
    bool preprocess(const cv::Mat& input_image, cv::Mat& clean_output, cv::Mat& binary_output);

    /**
     * @brief Configures cropping parameters to restrict processing to a centered ROI with offset.
     * @param width Width of the cropped region
     * @param height Height of the cropped region
     * @param offset_x Offset right from the center (default 0)
     * @param offset_y Offset down from the center (default 0)
     */
    void setCrop(int width, int height, int offset_x = 0, int offset_y = 0);

    /**
     * @brief Disables cropping, processing the entire input image.
     */
    void disableCrop();

    // Getters and Setters
    void setBgScale(float scale) { m_bg_scale = scale; }
    float getBgScale() const { return m_bg_scale; }

    void setSigma(float sigma) { m_sigma = sigma; }
    float getSigma() const { return m_sigma; }

    void setFiltSize(int filtsize) { m_filtsize = filtsize; }
    int getFiltSize() const { return m_filtsize; }

    bool isCropEnabled() const { return m_crop_enabled; }
    cv::Point getCropOffset() const { return cv::Point(m_crop_offset_x_actual, m_crop_offset_y_actual); }
};
