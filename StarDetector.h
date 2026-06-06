#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "identificator/DetectedStar.h"

class StarDetector {
private:
    float m_sigma_threshold;
    int m_min_area;

public:
    StarDetector(float sigma_threshold = 2.5f, int min_area = 2);
    std::vector<DetectedStar> process(const cv::Mat& image);
};
