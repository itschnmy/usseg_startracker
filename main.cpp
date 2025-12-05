#include "header.h"
int main() {
    // Test data
    // Body frame vectors
    std::vector<Eigen::Vector3d> bodyFrame;
    bodyFrame.push_back(Eigen::Vector3d(1, 0, 0));   // b1
    bodyFrame.push_back(Eigen::Vector3d(0, 1, 0));   // b2

    // Inertial frame vectors = Rz(90°) * body
    std::vector<Eigen::Vector3d> inertialFrame;
    inertialFrame.push_back(Eigen::Vector3d(0, 1, 0));    // r1
    inertialFrame.push_back(Eigen::Vector3d(-1, 0, 0));   // r2

    // Create estimators
    TRIADEstimator triad;
    QUESTEstimator quest;

    // Run TRIAD
    Eigen::Quaterniond q_triad = triad.estimate(bodyFrame, inertialFrame);
    std::cout << "TRIAD Quaternion (w,x,y,z): "
              << q_triad.w() << ", "
              << q_triad.x() << ", "
              << q_triad.y() << ", "
              << q_triad.z() << "\n";

    // Run QUEST
    Eigen::Quaterniond q_quest = quest.estimate(bodyFrame, inertialFrame);
    std::cout << "QUEST Quaternion (w,x,y,z): "
              << q_quest.w() << ", "
              << q_quest.x() << ", "
              << q_quest.y() << ", "
              << q_quest.z() << "\n";

    // ----------------------------------------------
    // Compare rotation matrices
    Eigen::Matrix3d R_triad = q_triad.toRotationMatrix();
    Eigen::Matrix3d R_quest = q_quest.toRotationMatrix();

    std::cout << "\nTRIAD Rotation Matrix:\n" << R_triad << "\n\n";
    std::cout << "QUEST Rotation Matrix:\n" << R_quest << "\n\n";

    // ----------------------------------------------
    // Optional: Check difference
    Eigen::Matrix3d diff = R_triad.transpose() * R_quest;
    std::cout << "Difference R_triad^T * R_quest:\n" << diff << "\n";

    return 0;
}