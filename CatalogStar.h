#pragma once
#include <Eigen/Dense>

struct CatalogStar {
    Eigen::Vector3d u; // unit vector in inertial frame
    float mag; //apparent magnitude
    double id; //star ID from Gaia DR3
};
