#pragma once
#include <Eigen/Dense>

struct CatalogStar {
    double id; //star id from Gaia DR3
    Eigen::Vector3d u; // unit vector of catalog star in inertial frame
    float magnitude; //apparent magnitude of the star
};