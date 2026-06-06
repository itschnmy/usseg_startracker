#pragma once
#include <Eigen/Dense>

struct DetectedStar {
    int index; //index of the detected blob in the image
    Eigen::Vector2d position; // 2D position in pixels
    double intensity; // sum of subtracted pixel intensities
    int peak; // maximum pixel intensity
    double radius; // approximate radius in pixels
    Eigen::Vector3d uBody; //unit vector in camera frame
};