#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#define ERROR 1e-12

// Skew (hat) operator
Eigen::Matrix3d hat(const Eigen::Vector3d& v);

// Unhat operator
Eigen::Vector3d unhat(const Eigen::Matrix3d& M);

// Build TRIAD orthonormal basis T = [t1 t2 t3]
Eigen::Matrix3d buildTriadBasis(
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    double eps = ERROR);

// Rotation matrix -> quaternion
Eigen::Quaterniond rot2q(const Eigen::Matrix3d& Q);

// body -> inertial
class AttitudeEstimator {
public:
    virtual Eigen::Quaterniond estimate(
        const std::vector<Eigen::Vector3d>& bodyFrame,
        const std::vector<Eigen::Vector3d>& inertialFrame) = 0; // pure virtual
    virtual ~AttitudeEstimator() = default;
};

class QUESTEstimator : public AttitudeEstimator {
public:
    Eigen::Quaterniond estimate(
        const std::vector<Eigen::Vector3d>& bodyFrame,
        const std::vector<Eigen::Vector3d>& inertialFrame) override;
};

class TRIADEstimator : public AttitudeEstimator {
public:
    Eigen::Quaterniond estimate(
        const std::vector<Eigen::Vector3d>& bodyFrame,
        const std::vector<Eigen::Vector3d>& inertialFrame) override;
};