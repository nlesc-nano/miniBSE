#pragma once
#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include "integrals_core.hpp"

namespace licpp {
    Eigen::MatrixXcd ao_ft_complex(const std::vector<libint2::Shell>& shells, const std::vector<KPoint>& kpts, int nthreads);
    Eigen::MatrixXd ao_values_at_points(const std::vector<libint2::Shell>& shells, const std::vector<Eigen::Vector3d>& points, int nthreads);
}

