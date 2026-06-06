#include "header.h"
#include "StarDetector.h"
#include "identificator/CameraModel.h"
#include "identificator/StarCatalog.h"
#include "identificator/PairCatalog.h"
#include "identificator/VotingIdentifier.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <Eigen/Geometry>
#include <string>

// Helper to print orientation in Euler angles (yaw, pitch, roll)
void printEuler(const Eigen::Quaterniond& q) {
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0); // ZYX
    std::cout << "Euler angles (Yaw, Pitch, Roll) in deg: " 
              << euler.x() * 180.0 / M_PI << ", "
              << euler.y() * 180.0 / M_PI << ", "
              << euler.z() * 180.0 / M_PI << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cout << "====================================================================\n";
        std::cout << "  STAR TRACKER PRODUCTION ATTITUDE ESTIMATOR\n";
        std::cout << "====================================================================\n";
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << " <image_path> <cx> <cy> <f> <star_catalog_csv> <kvector_db_bin>\n\n";
        std::cout << "Parameters:\n";
        std::cout << "  - image_path       : Path to the star field image (e.g., BMP, PNG)\n";
        std::cout << "  - cx               : Primary point X coordinate (pixels)\n";
        std::cout << "  - cy               : Primary point Y coordinate (pixels)\n";
        std::cout << "  - f                : Focal length (pixels)\n";
        std::cout << "  - star_catalog_csv : Path to the celestial star catalog CSV\n";
        std::cout << "  - kvector_db_bin   : Path to the binary K-Vector database\n";
        std::cout << "====================================================================\n";
        return 0;
    }

    std::string image_path = argv[1];
    double cx = std::stod(argv[2]);
    double cy = std::stod(argv[3]);
    double f = std::stod(argv[4]);
    std::string catalog_path = argv[5];
    std::string kvector_path = argv[6];

    std::cout << "Star Tracker initiated with parameters:\n";
    std::cout << "  - Image Path       : " << image_path << "\n";
    std::cout << "  - Center (cx, cy)  : (" << cx << ", " << cy << ")\n";
    std::cout << "  - Focal Length (f) : " << f << " pixels\n";
    std::cout << "  - Catalog Path     : " << catalog_path << "\n";
    std::cout << "  - K-Vector Path    : " << kvector_path << "\n\n";

    // 1. Load Star Catalog
    StarCatalog catalog;
    if (!catalog.loadFile(catalog_path)) {
        std::cerr << "Error: Failed to load star catalog CSV: " << catalog_path << "\n";
        return 1;
    }
    std::cout << "Loaded " << catalog.getStars().size() << " stars from catalog.\n";

    // 2. Load K-Vector Database
    PairCatalog kvector_db;
    if (!kvector_db.loadFile(kvector_path)) {
        std::cerr << "Error: Failed to load K-Vector database binary: " << kvector_path << "\n";
        return 1;
    }
    std::cout << "Loaded K-Vector DB (Min Dist: " << kvector_db.getMinDistance() 
              << " rad, Max Dist: " << kvector_db.getMaxDistance() << " rad).\n";

    // 3. Load Star Image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Failed to load image: " << image_path << "\n";
        return 1;
    }
    std::cout << "Loaded image: " << image.cols << " x " << image.rows << "\n";

    // Smooth image to reduce noise
    cv::Mat blurred_image;
    cv::GaussianBlur(image, blurred_image, cv::Size(3, 3), 0);

    // 4. Extract 2D Centroids (Hardware agnostic)
    StarDetector detector(2.5f, 2);
    std::vector<DetectedStar> detected_stars = detector.process(blurred_image);
    std::cout << "Detected " << detected_stars.size() << " star centroids in image.\n";

    if (detected_stars.empty()) {
        std::cerr << "Error: No stars detected in the image. Cannot solve attitude.\n";
        return 1;
    }

    // 5. Map 2D Coordinates to 3D Unit Vectors (Body frame) using CameraModel
    CameraModel camera(cx, cy, f);
    for (auto& star : detected_stars) {
        star.uBody = camera.pixelToUnitVector(star.position);
    }

    // 6. Identify Stars using Plate Solving (Voting Identifier)
    double tolerance = 0.02; // rad
    VotingIdentifier solver(catalog, kvector_db, tolerance);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> matched_vectors = solver.identify(detected_stars);
    std::cout << "VotingIdentifier identified " << matched_vectors.size() << " stars.\n";

    if (matched_vectors.size() < 2) {
        std::cerr << "Error: Plate solving failed. Identified " << matched_vectors.size() 
                  << " stars, but need at least 2 for attitude estimation.\n";
        return 1;
    }

    // 7. Estimate Attitude (QUEST)
    Eigen::Matrix3Xd bodyFrame(3, matched_vectors.size());
    Eigen::Matrix3Xd inertialFrame(3, matched_vectors.size());
    for (size_t i = 0; i < matched_vectors.size(); ++i) {
        bodyFrame.col(i) = matched_vectors[i].first;     // uBody
        inertialFrame.col(i) = matched_vectors[i].second; // uInertial
    }

    QUESTEstimator quest;
    try {
        Eigen::Quaterniond q_est = quest.estimate(bodyFrame, inertialFrame);
        Eigen::Matrix3d R_est = q_est.toRotationMatrix();

        std::cout << "\n====================================================\n";
        std::cout << "  ATTITUDE ESTIMATION SUCCESSFUL\n";
        std::cout << "====================================================\n";
        std::cout << "Estimated Rotation Matrix (R_est):\n" << R_est << "\n\n";
        std::cout << "Estimated Quaternion (w, x, y, z):\n" 
                  << "[" << q_est.w() << ", " << q_est.x() << ", " 
                  << q_est.y() << ", " << q_est.z() << "]\n\n";
        printEuler(q_est);
        std::cout << "====================================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Error in attitude estimation: " << e.what() << "\n";
        return 1;
    }

    return 0;
}