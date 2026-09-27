#include "CentroidMatcher.h"
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>

CentroidMatcher::CentroidMatcher(double dist_threshold, double inlier_threshold, int max_iterations)
    : m_dist_threshold(dist_threshold), m_inlier_threshold(inlier_threshold), m_max_iterations(max_iterations)
{}

// Helper to hash 2D grid coordinates
inline uint64_t getHashKey(int kx, int ky) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(kx)) << 32) | static_cast<uint32_t>(ky);
}

std::vector<std::pair<int, int>> CentroidMatcher::matchCentroids(
    const std::vector<Eigen::Vector2d>& prevCentroids,
    const std::vector<Eigen::Vector2d>& currCentroids
) {
    if (prevCentroids.empty() || currCentroids.empty()) {
        return {};
    }

    // 1. Spatial Hashing: Build grid for current centroids
    std::unordered_map<uint64_t, std::vector<int>> grid;
    for (size_t i = 0; i < currCentroids.size(); ++i) {
        int kx = static_cast<int>(std::floor(currCentroids[i].x() / m_dist_threshold));
        int ky = static_cast<int>(std::floor(currCentroids[i].y() / m_dist_threshold));
        grid[getHashKey(kx, ky)].push_back(static_cast<int>(i));
    }

    // 2. Local Search: Find all candidate matches within dist_threshold
    std::vector<std::pair<int, int>> candidatePairs;
    for (size_t i = 0; i < prevCentroids.size(); ++i) {
        int p_kx = static_cast<int>(std::floor(prevCentroids[i].x() / m_dist_threshold));
        int p_ky = static_cast<int>(std::floor(prevCentroids[i].y() / m_dist_threshold));

        // Search p_kx-1 to p_kx+1, p_ky-1 to p_ky+1 (9 cells total)
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                uint64_t key = getHashKey(p_kx + dx, p_ky + dy);
                auto it = grid.find(key);
                if (it != grid.end()) {
                    for (int j : it->second) {
                        double dist = (prevCentroids[i] - currCentroids[j]).norm();
                        if (dist <= m_dist_threshold) {
                            candidatePairs.push_back({static_cast<int>(i), j});
                        }
                    }
                }
            }
        }
    }

    if (candidatePairs.size() < 2) {
        return candidatePairs; // Too few candidate matches to run RANSAC
    }

    // 3. RANSAC Loop
    std::mt19937 rng(1337); // Seeded random number generator
    std::uniform_int_distribution<size_t> dist_rand(0, candidatePairs.size() - 1);

    std::vector<int> bestInliers;
    Eigen::Matrix2d bestR = Eigen::Matrix2d::Identity();
    Eigen::Vector2d bestT = Eigen::Vector2d::Zero();

    for (int iter = 0; iter < m_max_iterations; ++iter) {
        // Randomly select 2 candidate pairs
        size_t idx1 = dist_rand(rng);
        size_t idx2 = dist_rand(rng);
        if (idx1 == idx2) continue;

        auto pair1 = candidatePairs[idx1];
        auto pair2 = candidatePairs[idx2];

        // Ensure the two points in prevCentroids are distinct enough
        Eigen::Vector2d p1 = prevCentroids[pair1.first];
        Eigen::Vector2d p2 = prevCentroids[pair2.first];
        if ((p2 - p1).norm() < 10.0) continue; // Skip to avoid numerical instability

        Eigen::Vector2d p1_prime = currCentroids[pair1.second];
        Eigen::Vector2d p2_prime = currCentroids[pair2.second];

        // Solve for rotation & translation (2-point solver)
        Eigen::Vector2d v = p2 - p1;
        Eigen::Vector2d v_prime = p2_prime - p1_prime;

        double theta = std::atan2(v_prime.y(), v_prime.x()) - std::atan2(v.y(), v.x());
        Eigen::Matrix2d R;
        R << std::cos(theta), -std::sin(theta),
             std::sin(theta),  std::cos(theta);

        Eigen::Vector2d t = 0.5 * (p1_prime + p2_prime) - R * (0.5 * (p1 + p2));

        // Evaluate model inliers
        std::vector<int> currentInliers;
        for (size_t i = 0; i < candidatePairs.size(); ++i) {
            int prev_idx = candidatePairs[i].first;
            int curr_idx = candidatePairs[i].second;

            Eigen::Vector2d projected = R * prevCentroids[prev_idx] + t;
            double error = (currCentroids[curr_idx] - projected).norm();

            if (error <= m_inlier_threshold) {
                currentInliers.push_back(static_cast<int>(i));
            }
        }

        if (currentInliers.size() > bestInliers.size()) {
            bestInliers = currentInliers;
            bestR = R;
            bestT = t;
        }
    }

    // 4. Refinement step using all inliers in the best set
    if (bestInliers.size() >= 2) {
        Eigen::Vector2d prevMean = Eigen::Vector2d::Zero();
        Eigen::Vector2d currMean = Eigen::Vector2d::Zero();
        for (int idx : bestInliers) {
            prevMean += prevCentroids[candidatePairs[idx].first];
            currMean += currCentroids[candidatePairs[idx].second];
        }
        prevMean /= static_cast<double>(bestInliers.size());
        currMean /= static_cast<double>(bestInliers.size());

        double s_xx = 0, s_xy = 0, s_yx = 0, s_yy = 0;
        for (int idx : bestInliers) {
            Eigen::Vector2d q = prevCentroids[candidatePairs[idx].first] - prevMean;
            Eigen::Vector2d q_prime = currCentroids[candidatePairs[idx].second] - currMean;

            s_xx += q.x() * q_prime.x();
            s_xy += q.x() * q_prime.y();
            s_yx += q.y() * q_prime.x();
            s_yy += q.y() * q_prime.y();
        }

        double refinedTheta = std::atan2(s_xy - s_yx, s_xx + s_yy);
        bestR << std::cos(refinedTheta), -std::sin(refinedTheta),
                 std::sin(refinedTheta),  std::cos(refinedTheta);
        bestT = currMean - bestR * prevMean;
    }

    // 5. Select final matched pairs based on refined inliers
    std::vector<std::pair<int, int>> finalMatches;
    std::vector<bool> prevMatched(prevCentroids.size(), false);
    std::vector<bool> currMatched(currCentroids.size(), false);

    for (int idx : bestInliers) {
        int prev_idx = candidatePairs[idx].first;
        int curr_idx = candidatePairs[idx].second;

        // Double check refined projection error
        Eigen::Vector2d projected = bestR * prevCentroids[prev_idx] + bestT;
        double error = (currCentroids[curr_idx] - projected).norm();

        if (error <= m_inlier_threshold && !prevMatched[prev_idx] && !currMatched[curr_idx]) {
            finalMatches.push_back({prev_idx, curr_idx});
            prevMatched[prev_idx] = true;
            currMatched[curr_idx] = true;
        }
    }

    return finalMatches;
}
