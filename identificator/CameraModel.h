#pragma once
#include <Eigen/Dense>

class CameraModel {
private:
    double m_cx;
    double m_cy;
    double m_f;

public:
    CameraModel(double cx, double cy, double f) 
        : m_cx(cx), m_cy(cy), m_f(f) {}
        
    Eigen::Vector3d pixelToUnitVector(const Eigen::Vector2d& pixel) const {
        double x_norm = (pixel.x() - m_cx) / m_f;
        double y_norm = (pixel.y() - m_cy) / m_f;
        Eigen::Vector3d u(x_norm, y_norm, 1.0);
        return u.normalized();
    }

    double getCx() const { return m_cx; }
    double getCy() const { return m_cy; }
    double getF() const { return m_f; }
};
