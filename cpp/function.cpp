#include "header.h"

Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
    Eigen::Matrix3d M;
    M <<  0.0,   -v.z(),  v.y(),
          v.z(),  0.0,   -v.x(),
         -v.y(),  v.x(),  0.0;
    return M;
}

Eigen::Vector3d unhat(const Eigen::Matrix3d& M) {
    return Eigen::Vector3d(
        -M(1,2),
         M(0,2),
        -M(0,1)
    );
}

Eigen::Matrix3d buildTriadBasis(
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    double eps)
{
    Eigen::Vector3d t1 = v1.normalized();

    Eigen::Vector3d c12 = v1.cross(v2);
    double n12 = c12.norm();
    if (n12 < eps) {
        throw std::runtime_error("buildTriadBasis: input vectors are nearly collinear.");
    }
    Eigen::Vector3d t2 = c12 / n12;

    Eigen::Vector3d c13 = t1.cross(t2);
    double n13 = c13.norm();
    if (n13 < eps) {
        throw std::runtime_error("buildTriadBasis: degenerate basis while building t3.");
    }
    Eigen::Vector3d t3 = c13 / n13;

    Eigen::Matrix3d T;
    T.col(0) = t1;
    T.col(1) = t2;
    T.col(2) = t3;
    return T;
}

Eigen::Quaterniond rot2q(const Eigen::Matrix3d& Q) {
    Eigen::AngleAxisd aa(Q);
    double theta = aa.angle();

    if (std::abs(theta) < 1e-15) {
        return Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0); // identity
    }

    Eigen::Vector3d axis = aa.axis().normalized();
    double s = std::sin(theta * 0.5);
    double c = std::cos(theta * 0.5);

    Eigen::Quaterniond q(c, axis.x() * s, axis.y() * s, axis.z() * s);
    q.normalize();
    return q;
}