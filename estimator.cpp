#include "header.h"

Eigen::Quaterniond TRIADEstimator::estimate(
    const Eigen::Matrix3Xd& bodyFrame,
    const Eigen::Matrix3Xd& inertialFrame)
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

    return rot2q(Q);
}

Eigen::Quaterniond QUESTEstimator::estimate(
    const Eigen::Matrix3Xd& bodyFrame,
    const Eigen::Matrix3Xd& inertialFrame)
{
    const size_t N = bodyFrame.cols();
    if (N < 2 || inertialFrame.cols() != N) {
        throw std::runtime_error("QUEST estimate requires N >= 2 matching vector pairs.");
    }

    if (N == 2) {
        TRIADEstimator triad;
        return triad.estimate(bodyFrame, inertialFrame);
    }

    // .noalias() prevents Eigen from creating a temporary hidden matrix.
    Eigen::Matrix3d B;
    B.noalias() = inertialFrame * bodyFrame.transpose();

    const double sigma = B.trace();
    const Eigen::Matrix3d S = B + B.transpose();

    Eigen::Vector3d z;
    z << (B(1,2) - B(2,1)),
         (B(2,0) - B(0,2)),
         (B(0,1) - B(1,0));

    // QUEST ALGORITHM
    
    // 1. Calculate coefficients for the characteristic polynomial
    const double kappa = 0.5 * (S.trace() * S.trace() - (S * S).trace());
    const double delta = S.determinant();
    const double a = sigma * sigma - kappa;
    const double b = sigma * sigma + z.squaredNorm();
    const double c = delta + z.dot(S * z);
    const double d = z.dot(S * S * z);

    // 2. Newton-Raphson iteration for the maximum eigenvalue (lambda_max)
    // Initial guess is the sum of the weights (which is N for unweighted vectors)
    double lambda = static_cast<double>(N); 

    for (int i = 0; i < 4; ++i) { // Usually converges in 1-2 iterations
        double lambda2 = lambda * lambda;
        
        // f(lambda) and f'(lambda)
        double f = lambda2 * lambda2 - (a + b) * lambda2 - c * lambda + (a * b + c * sigma - d);
        double f_prime = 4.0 * lambda * lambda2 - 2.0 * (a + b) * lambda - c;
        
        if (std::abs(f_prime) < ERROR) break;
        
        double step = f / f_prime;
        lambda -= step;
        
        if (std::abs(step) < ERROR) break;
    }

    // 3. Calculate Rodrigues parameters (Gibbs vector)
    Eigen::Matrix3d denom = (lambda + sigma) * Eigen::Matrix3d::Identity() - S;
    
    // Note: If facing a near 180-degree rotation, denom.determinant() approaches 0.
    Eigen::Vector3d p = denom.inverse() * z;

    // 4. Convert Gibbs vector directly to Quaternion
    double factor = 1.0 / std::sqrt(1.0 + p.squaredNorm());
    
    // Eigen::Quaterniond constructor is (w, x, y, z)
    return Eigen::Quaterniond(factor, p.x() * factor, p.y() * factor, p.z() * factor);
}
