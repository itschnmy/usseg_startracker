#pragma once
#include <Eigen/Dense>

struct DetectedStar {
    int index; //index of the detected blob in the image
    Eigen::Vector3d uBody; //unit vector in camera frame
};