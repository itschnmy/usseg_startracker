// =============================================================================
// TEST: LIS (Lost-In-Space) Mode — Attitude Determination Pipeline
// =============================================================================
// Pipeline tested (ref: Research Log/images/image14.png):
//   [Star Catalog + K-Vector DB]
//     -> Vector Generator (CameraModel: pixel -> uBody)
//       -> Star Identificator (VotingIdentifier: uBody pairs -> catalog match)
//         -> Attitude Determinator (QUEST: bodyFrame + inertialFrame -> quaternion)
//
// OpenCV is NOT required. Star centroids are generated mathematically
// by projecting known catalog stars through a simulated camera orientation.
// =============================================================================

#include "header.h"
#include "identificator/CameraModel.h"
#include "identificator/DetectedStar.h"
#include "identificator/StarCatalog.h"
#include "identificator/PairCatalog.h"
#include "identificator/VotingIdentifier.h"

#include <iostream>
#include <Eigen/Geometry>
#include <algorithm>
#include <iomanip>
#include <cmath>

void printEuler(const Eigen::Quaterniond& q) {
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0); // ZYX
    std::cout << "Euler angles (Yaw, Pitch, Roll) in deg: " 
              << euler.x() * 180.0 / M_PI << ", "
              << euler.y() * 180.0 / M_PI << ", "
              << euler.z() * 180.0 / M_PI << "\n";
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "  TEST: LIS MODE — ATTITUDE DETERMINATION\n";
    std::cout << "  (No OpenCV — pure algorithmic verification)\n";
    std::cout << "==================================================\n\n";

    // -----------------------------------------------------------------
    // SURVEYED CAMERA PARAMETERS (from test/img BMP analysis)
    // -----------------------------------------------------------------
    const double cx = 768.0;
    const double cy = 1024.0;
    const double f = 1313.15;
    const int img_width = 1536;
    const int img_height = 2048;

    std::cout << "Camera parameters:\n";
    std::cout << "  Resolution : " << img_width << " x " << img_height << "\n";
    std::cout << "  Center     : (" << cx << ", " << cy << ")\n";
    std::cout << "  Focal len  : " << f << " px\n\n";

    CameraModel camera(cx, cy, f);

    // -----------------------------------------------------------------
    // 1. LOAD STAR CATALOG & K-VECTOR DATABASE
    // -----------------------------------------------------------------
    StarCatalog catalog;
    if (!catalog.loadFile("star_catalog.csv")) {
        std::cerr << "FAIL: Cannot load star_catalog.csv\n";
        return 1;
    }
    std::cout << "Loaded " << catalog.getStars().size() << " catalog stars.\n";

    PairCatalog kvector_db;
    if (!kvector_db.loadFile("kvector_fixed.db")) {
        std::cerr << "FAIL: Cannot load kvector_fixed.db\n";
        return 1;
    }
    std::cout << "Loaded K-Vector DB (range: [" << kvector_db.getMinDistance() 
              << ", " << kvector_db.getMaxDistance() << "] rad).\n";

    // -----------------------------------------------------------------
    // 2. SELECT TEST STARS FROM CATALOG (Southern Cross / Crux)
    // -----------------------------------------------------------------
    std::vector<CatalogStar> all_catalog = catalog.getStars();
    std::vector<CatalogStar> crux_stars;
    std::vector<int> crux_ids = {60718, 62434, 61084, 59747, 60260};
    for (const auto& star : all_catalog) {
        if (std::find(crux_ids.begin(), crux_ids.end(), static_cast<int>(star.id)) != crux_ids.end()) {
            crux_stars.push_back(star);
        }
    }
    if (crux_stars.size() < 4) {
        std::cerr << "Warning: Crux stars not found. Using first 5 catalog stars.\n";
        crux_stars.clear();
        for (size_t i = 0; i < 5 && i < all_catalog.size(); ++i)
            crux_stars.push_back(all_catalog[i]);
    }
    std::cout << "Selected " << crux_stars.size() << " test stars.\n";

    // -----------------------------------------------------------------
    // 3. DEFINE TRUE ATTITUDE & PROJECT STARS TO PIXEL COORDINATES
    //    (This replaces the OpenCV image simulation)
    // -----------------------------------------------------------------
    // Calculate the mean vector of the crux stars to point the camera at them
    Eigen::Vector3d mean_u = Eigen::Vector3d::Zero();
    for (const auto& star : crux_stars) {
        mean_u += star.u;
    }
    mean_u.normalize();

    // Construct a camera orientation R_true such that the camera Z-axis points at mean_u
    Eigen::Vector3d z_body = mean_u;
    Eigen::Vector3d temp = (std::abs(z_body.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    Eigen::Vector3d x_body = temp.cross(z_body).normalized();
    Eigen::Vector3d y_body = z_body.cross(x_body).normalized();

    Eigen::Matrix3d R_true;
    R_true.col(0) = x_body;
    R_true.col(1) = y_body;
    R_true.col(2) = z_body;

    Eigen::Quaterniond q_true(R_true);
    std::cout << "\nTrue orientation (R):\n" << R_true << "\n";
    printEuler(q_true);

    // Mathematically project catalog stars to 2D pixel coordinates
    // then construct DetectedStar structs directly (no image needed)
    std::vector<DetectedStar> detected;
    int star_id = 0;
    for (const auto& star : crux_stars) {
        Eigen::Vector3d uBody = R_true.transpose() * star.u;
        if (uBody.z() > 0) {
            double x_pixel = (uBody.x() / uBody.z()) * f + cx;
            double y_pixel = (uBody.y() / uBody.z()) * f + cy;

            if (x_pixel >= 0 && x_pixel < img_width && y_pixel >= 0 && y_pixel < img_height) {
                DetectedStar ds;
                ds.index = star_id++;
                ds.position = Eigen::Vector2d(x_pixel, y_pixel);
                ds.intensity = 1000.0;  // placeholder
                ds.peak = 255;          // placeholder
                ds.radius = 3.0;        // placeholder
                ds.uBody = camera.pixelToUnitVector(ds.position);
                detected.push_back(ds);
            }
        }
    }
    std::cout << "Projected " << detected.size() << " stars to pixel coordinates.\n";

    // -----------------------------------------------------------------
    // 4. STAR IDENTIFICATION (VotingIdentifier + K-Vector Plate Solving)
    // -----------------------------------------------------------------
    double tolerance = 0.001; // rad
    VotingIdentifier solver(catalog, kvector_db, tolerance);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> matched = solver.identify(detected);
    std::cout << "VotingIdentifier matched " << matched.size() << " star pairs.\n";

    if (matched.size() < 2) {
        std::cerr << ">>> LIS Mode: FAILED (could not identify enough stars) <<<\n";
        return 1;
    }

    // -----------------------------------------------------------------
    // 5. ATTITUDE DETERMINATION (QUEST)
    // -----------------------------------------------------------------
    Eigen::Matrix3Xd bodyFrame(3, matched.size());
    Eigen::Matrix3Xd inertialFrame(3, matched.size());
    for (size_t i = 0; i < matched.size(); ++i) {
        bodyFrame.col(i) = matched[i].first;
        inertialFrame.col(i) = matched[i].second;
    }

    QUESTEstimator quest;
    Eigen::Quaterniond q_est = quest.estimate(bodyFrame, inertialFrame);
    Eigen::Matrix3d R_est = q_est.toRotationMatrix();

    std::cout << "\nQUEST estimated orientation (R_est):\n" << R_est << "\n";
    printEuler(q_est);

    std::cout << "\nMatch Details:\n";
    for (size_t i = 0; i < matched.size(); ++i) {
        int matched_cat_id = -1;
        for (const auto& star : all_catalog) {
            if ((star.u - matched[i].second).norm() < 1e-6) {
                matched_cat_id = star.id;
                break;
            }
        }
        
        int true_cat_id = -1;
        for (const auto& star : crux_stars) {
            Eigen::Vector3d expected_uBody = R_true.transpose() * star.u;
            if ((expected_uBody - matched[i].first).norm() < 1e-4) {
                true_cat_id = star.id;
                break;
            }
        }
        std::cout << "  Detected Star " << i << " (pos: [" << detected[i].position.x() << ", " << detected[i].position.y() 
                  << "]) -> Matched Catalog ID: " << matched_cat_id 
                  << " | True Catalog ID: " << true_cat_id 
                  << (matched_cat_id == true_cat_id ? " [CORRECT]" : " [WRONG]") << "\n";
    }

    // -----------------------------------------------------------------
    // 6. VERIFICATION: Compare estimated vs true attitude
    //    Note: QUEST estimates the passive rotation R_est (inertial to body),
    //    which is the transpose of the active rotation R_true (body to inertial).
    //    Therefore, R_true * R_est should be the identity matrix.
    // -----------------------------------------------------------------
    Eigen::Matrix3d diff = R_true * R_est;
    double error_angle = Eigen::AngleAxisd(diff).angle() * 180.0 / M_PI;
    std::cout << "\nOrientation Error: " << error_angle << " deg\n";

    if (error_angle < 0.1) {
        std::cout << ">>> LIS Mode Verification: SUCCESS! <<<\n";
        return 0;
    } else {
        std::cout << ">>> LIS Mode Verification: FAILED (error = " << error_angle << " deg) <<<\n";
        return 1;
    }
}
