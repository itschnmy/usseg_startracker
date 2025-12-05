#include "VotingIdentifier.h"
#include <iostream>
#include <unordered_map>

VotingIdentifier::VotingIdentifier(StarCatalog& scat, PairCatalog& pcat, double angleTol)
    : starCatalog(scat), pairCatalog(pcat), angleTol(angleTol) 
{}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> VotingIdentifier::identify(std::vector<DetectedStar> detected) {
}