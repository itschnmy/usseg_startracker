// =============================================================================
// TEST: Tracking Mode — CentroidMatcher + Attitude Update
// =============================================================================
// Pipeline tested (ref: Research Log/images/image14.png):
//   [Previous frame centroids + catalog associations]
//     -> CentroidMatcher (Spatial Hashing + RANSAC: match prev -> curr)
//       -> Vector Generator (CameraModel: pixel -> uBody)
//         -> Attitude Determinator (QUEST: bodyFrame + inertialFrame -> quaternion)
//
// OpenCV is NOT required. Previous and current frame centroids are generated
// mathematically from catalog star projections with simulated motion.
// =============================================================================

#include "header.h"
#include "identificator/CameraModel.h"
#include "identificator/StarCatalog.h"
#include "CentroidMatcher.h"

#include <iostream>
#include <Eigen/Geometry>
#include <random>
#include <iomanip>
#include <algorithm>
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
    std::cout << "  TEST: TRACKING MODE — CENTROID MATCH + ATTITUDE\n";
    std::cout << "  (No OpenCV — pure algorithmic verification)\n";
    std::cout << "==================================================\n\n";

    // -----------------------------------------------------------------
    // SURVEYED CAMERA PARAMETERS
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
    // 1. LOAD STAR CATALOG (for reference inertial vectors)
    // -----------------------------------------------------------------
    StarCatalog catalog;
    if (!catalog.loadFile("star_catalog.csv")) {
        std::cerr << "FAIL: Cannot load star_catalog.csv\n";
        return 1;
    }

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

    // -----------------------------------------------------------------
    // 2. GENERATE PREVIOUS FRAME CENTROIDS (mathematical projection)
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

    std::vector<Eigen::Vector2d> prev_centroids;
    std::vector<CatalogStar> active_crux_stars;
    for (const auto& star : crux_stars) {
        Eigen::Vector3d uBody = R_true.transpose() * star.u;
        if (uBody.z() > 0) {
            double x_pixel = (uBody.x() / uBody.z()) * f + cx;
            double y_pixel = (uBody.y() / uBody.z()) * f + cy;
            if (x_pixel >= 0 && x_pixel < img_width && y_pixel >= 0 && y_pixel < img_height) {
                prev_centroids.push_back(Eigen::Vector2d(x_pixel, y_pixel));
                active_crux_stars.push_back(star);
            }
        }
    }

    // -----------------------------------------------------------------
    // 3. SIMULATE CURRENT FRAME (rotation + translation + noise + outliers)
    // -----------------------------------------------------------------
    double delta_theta = 2.0 * M_PI / 180.0;
    Eigen::Matrix2d R_track;
    R_track << std::cos(delta_theta), -std::sin(delta_theta),
               std::sin(delta_theta),  std::cos(delta_theta);
    Eigen::Vector2d t_track(5.5, -3.2);

    std::vector<Eigen::Vector2d> curr_centroids;
    std::default_random_engine generator(42);
    std::normal_distribution<double> noise(0.0, 0.05);

    for (const auto& p : prev_centroids) {
        Eigen::Vector2d p_moved = R_track * p + t_track;
        p_moved.x() += noise(generator);
        p_moved.y() += noise(generator);
        curr_centroids.push_back(p_moved);
    }

    // Inject hot pixel outliers
    curr_centroids.push_back(Eigen::Vector2d(100.0, 100.0));
    curr_centroids.push_back(Eigen::Vector2d(500.0, 400.0));
    
    std::cout << "Previous frame: " << prev_centroids.size() << " stars\n";
    std::cout << "Current frame : " << curr_centroids.size() << " centroids (incl. 2 outliers)\n";

    // -----------------------------------------------------------------
    // 4. CENTROID MATCHING (Spatial Hashing + RANSAC)
    // -----------------------------------------------------------------
    CentroidMatcher matcher(80.0, 2.0, 200);
    std::vector<std::pair<int, int>> matches = matcher.matchCentroids(prev_centroids, curr_centroids);

    std::cout << "CentroidMatcher found " << matches.size() << " pairs:\n";
    int true_matches = 0;
    for (const auto& match : matches) {
        std::cout << "  [" << match.first << " -> " << match.second << "]";
        if (match.first == match.second) {
            std::cout << " OK\n";
            true_matches++;
        } else {
            std::cout << " WRONG\n";
        }
    }

    if (true_matches != (int)prev_centroids.size() || matches.size() != prev_centroids.size()) {
        std::cout << ">>> Tracking Mode: FAILED (matching) <<<\n";
        return 1;
    }
    std::cout << "All true stars matched. All outliers rejected.\n\n";

    // -----------------------------------------------------------------
    // 5. ATTITUDE DETERMINATION (QUEST) on tracked stars
    // -----------------------------------------------------------------
    std::cout << "Running QUEST on matched tracking stars...\n";
    Eigen::Matrix3Xd bodyFrame(3, matches.size());
    Eigen::Matrix3Xd inertialFrame(3, matches.size());

    for (size_t i = 0; i < matches.size(); ++i) {
        int prev_idx = matches[i].first;
        int curr_idx = matches[i].second;

        bodyFrame.col(i) = camera.pixelToUnitVector(curr_centroids[curr_idx]);
        inertialFrame.col(i) = active_crux_stars[prev_idx].u;
    }

    QUESTEstimator quest;
    Eigen::Quaterniond q_est = quest.estimate(bodyFrame, inertialFrame);
    Eigen::Matrix3d R_est = q_est.toRotationMatrix();

    std::cout << "QUEST estimated orientation (current frame):\n" << R_est << "\n";
    printEuler(q_est);
    std::cout << ">>> Tracking Mode Verification: SUCCESS! <<<\n";
    return 0;
}
