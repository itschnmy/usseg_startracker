#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct KVectorPair {
    int16_t index1;
    int16_t index2;
};

class PairCatalog {
private:
    int32_t magic;
    int32_t num_pairs;
    float min_distance;
    float max_distance;
    int32_t num_bins;
    std::vector<KVectorPair> pairs;
    std::vector<int32_t> bins;

    int binForDistance(double distance) const;

public:
    bool loadFile(std::string path); //load binary k-vector database
    std::vector<KVectorPair> queryPairs(double minDistance, double maxDistance) const;

    double getMinDistance() const { return min_distance; }
    double getMaxDistance() const { return max_distance; }
};