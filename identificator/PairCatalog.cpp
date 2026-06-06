#include "PairCatalog.h"
#include <fstream>
#include <iostream>
#include <cmath>

#define K_VECTOR_MAGIC_NUMBER 0x4253F009

bool PairCatalog::loadFile(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open binary k-vector file: " << path << std::endl;
        return false;
    }

    // Read header: magic, num_pairs, min_distance, max_distance, num_bins
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != K_VECTOR_MAGIC_NUMBER) {
        std::cerr << "Error: Invalid k-vector database magic number: " << std::hex << magic << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&num_pairs), sizeof(num_pairs));
    file.read(reinterpret_cast<char*>(&min_distance), sizeof(min_distance));
    file.read(reinterpret_cast<char*>(&max_distance), sizeof(max_distance));
    file.read(reinterpret_cast<char*>(&num_bins), sizeof(num_bins));

    if (min_distance < 0.0f || max_distance <= min_distance || num_bins <= 0) {
        std::cerr << "Error: Invalid k-vector database header parameters." << std::endl;
        return false;
    }

    // Read pairs
    pairs.resize(num_pairs);
    file.read(reinterpret_cast<char*>(pairs.data()), num_pairs * sizeof(KVectorPair));

    // Read bins
    bins.resize(num_bins + 1);
    file.read(reinterpret_cast<char*>(bins.data()), (num_bins + 1) * sizeof(int32_t));

    file.close();
    return true;
}

int PairCatalog::binForDistance(double distance) const {
    double bin_width = (max_distance - min_distance) / num_bins;
    int result = static_cast<int>(std::floor((distance - min_distance) / bin_width));
    if (result < 0) return 0;
    if (result >= num_bins) return num_bins - 1;
    return result;
}

std::vector<KVectorPair> PairCatalog::queryPairs(double minDistance, double maxDistance) const {
    if (maxDistance <= minDistance) {
        return {};
    }

    if (minDistance < min_distance || minDistance > max_distance ||
        maxDistance < min_distance || maxDistance > max_distance) {
        return {};
    }

    int lower_bin = binForDistance(minDistance);
    int upper_bin = binForDistance(maxDistance);

    int32_t lower_pair = bins[lower_bin];
    int32_t upper_pair = bins[upper_bin + 1];

    if (lower_pair >= upper_pair || lower_pair < 0 || upper_pair > num_pairs) {
        return {};
    }

    return std::vector<KVectorPair>(pairs.begin() + lower_pair, pairs.begin() + upper_pair);
}