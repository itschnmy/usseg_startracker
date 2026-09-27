#pragma once
#include <vector>
#include <Eigen/Dense>
#include <utility>

class CentroidMatcher {
private:
    double m_dist_threshold;
    double m_inlier_threshold;
    int m_max_iterations;

public:
    CentroidMatcher(double dist_threshold = 40.0, double inlier_threshold = 2.0, int max_iterations = 200);

    // Returns a list of matched pairs of indices: std::pair<prev_index, curr_index>
    std::vector<std::pair<int, int>> matchCentroids(
        const std::vector<Eigen::Vector2d>& prevCentroids,
        const std::vector<Eigen::Vector2d>& currCentroids
    );
};
