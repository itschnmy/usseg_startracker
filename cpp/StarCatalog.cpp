#include "StarCatalog.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

bool StarCatalog::loadFile(std::string path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open star catalog file: " << path << std::endl;
        return false;
    }

    stars.clear();
    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        return false;
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        
        double id = 0.0;
        double ux = 0.0, uy = 0.0, uz = 0.0;
        float magnitude = 0.0f;

        try {
            if (!std::getline(ss, val, ',')) continue;
            id = std::stod(val);

            if (!std::getline(ss, val, ',')) continue;
            ux = std::stod(val);

            if (!std::getline(ss, val, ',')) continue;
            uy = std::stod(val);

            if (!std::getline(ss, val, ',')) continue;
            uz = std::stod(val);

            if (!std::getline(ss, val, ',')) continue;
            magnitude = std::stof(val);

            CatalogStar star;
            star.id = id;
            star.u = Eigen::Vector3d(ux, uy, uz);
            star.magnitude = magnitude;
            stars.push_back(star);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << ", error: " << e.what() << std::endl;
            continue;
        }
    }
    file.close();
    return true;
}

std::vector<CatalogStar> StarCatalog::getStars() {
    return stars;
}