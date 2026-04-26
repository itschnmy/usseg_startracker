#include "header.h"
int main() {
    AttitudeControlSystem adcs;
    // TODO: Read the JSON or binary struct generated.
    // For demonstration, simulating passing N vectors:
    
    int num_vectors = 4; // Simulated input
    Eigen::Matrix3Xd body_matrix(3, num_vectors);
    Eigen::Matrix3Xd inertial_matrix(3, num_vectors);
    
    // Fill matrices with translated data...
    // body_matrix.col(0) = Eigen::Vector3d(x, y, z); 
    
    try {
        Eigen::Quaterniond attitude = adcs.processSensorData(body_matrix, inertial_matrix);
        std::cout << "Final Spacecraft Attitude (w,x,y,z): "
                  << attitude.w() << ", " << attitude.x() << ", "
                  << attitude.y() << ", " << attitude.z() << "\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        // Command safe mode routine
    }
    return 0;
}
