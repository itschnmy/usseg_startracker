#include "header.h"

Eigen::Quaterniond TRIADEstimator::estimate(
    const std::vector<Eigen::Vector3d>& bodyFrame,
    const std::vector<Eigen::Vector3d>& inertialFrame)
{
    if (bodyFrame.size() < 2 || inertialFrame.size() < 2) {
        throw std::runtime_error("TRIAD estimate requires at least 2 vector pairs.");
    }

    // N is inertial, B is body 
    const Eigen::Vector3d& rN1 = inertialFrame[0];
    const Eigen::Vector3d& rN2 = inertialFrame[1];
    const Eigen::Vector3d& rB1 = bodyFrame[0];
    const Eigen::Vector3d& rB2 = bodyFrame[1];

    // Tn = [tN1 tN2 cross(tN1,tN2)/norm(...)]
    // Tb = [tB1 tB2 cross(tB1,tB2)/norm(...)]
    Eigen::Matrix3d Tn = buildTriadBasis(rN1, rN2);
    Eigen::Matrix3d Tb = buildTriadBasis(rB1, rB2);

    // Q = Tn*Tb';
    Eigen::Matrix3d Q = Tn * Tb.transpose();

    // Matrix -> Quaterniond
    Eigen::Quaterniond q = rot2q(Q);
    q.normalize();

    return q;
}

Eigen::Quaterniond QUESTEstimator::estimate(
    const std::vector<Eigen::Vector3d>& bodyFrame,
    const std::vector<Eigen::Vector3d>& inertialFrame)
{
    const size_t N = bodyFrame.size();
    if (N < 2 || inertialFrame.size() != N) {
        throw std::runtime_error("QUEST estimate requires N >= 2 matching vector pairs.");
    }

    // TRIAD Fallback if N == 2
    if (N == 2) {
        const Eigen::Vector3d& rN1 = inertialFrame[0];
        const Eigen::Vector3d& rN2 = inertialFrame[1];
        const Eigen::Vector3d& rB1 = bodyFrame[0];
        const Eigen::Vector3d& rB2 = bodyFrame[1];

        Eigen::Matrix3d Tn = buildTriadBasis(rN1, rN2);
        Eigen::Matrix3d Tb = buildTriadBasis(rB1, rB2);
        Eigen::Matrix3d Q  = Tn * Tb.transpose();

        Eigen::Quaterniond q = rot2q(Q);
        q.normalize();
        return q;
    }
    // Use B = Σ ( rN_i * rB_i^T ) to match with TRIAD 
    Eigen::Matrix3d B = Eigen::Matrix3d::Zero();

    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d rB = bodyFrame[i].normalized();
        Eigen::Vector3d rN = inertialFrame[i].normalized();
        B += rN * rB.transpose();
    }

    const double sigma = B.trace();
    const Eigen::Matrix3d S = B + B.transpose();

    Eigen::Vector3d z;
    z << (B(1,2) - B(2,1)),
         (B(2,0) - B(0,2)),
         (B(0,1) - B(1,0));

    // Davenport K matrix
    Eigen::Matrix4d K = Eigen::Matrix4d::Zero();
    K.block<3,3>(0,0) = S - sigma * Eigen::Matrix3d::Identity();
    K.block<3,1>(0,3) = z;
    K.block<1,3>(3,0) = z.transpose();
    K(3,3) = sigma;

    // Solve for max eigenvector
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eig(K);
    if (eig.info() != Eigen::Success) {
        throw std::runtime_error("QUEST: eigen decomposition failed.");
    }

    int idx = 0;
    eig.eigenvalues().maxCoeff(&idx);
    Eigen::Vector4d qv = eig.eigenvectors().col(idx);

    // Convert to Eigen::Quaterniond(w, x, y, z).
    Eigen::Quaterniond q(qv(3), qv(0), qv(1), qv(2));
    q.normalize();

    return q;
}