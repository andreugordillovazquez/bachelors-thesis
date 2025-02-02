#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <execution> // For parallelism
#include <numeric>   // For efficient reductions

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Helper function to generate Gauss-Hermite quadrature nodes and weights
void GaussHermite_Locations_Weights(int N, std::vector<double>& nodes, std::vector<double>& weights) {
    if (N == 2) {
        nodes = {-0.7071067811865475, 0.7071067811865475};
        weights = {0.8862269254527580, 0.8862269254527580};
    }
}

// Generate PAM constellation
std::vector<double> generatePAMConstellation(int M, double d) {
    std::vector<double> X(M);
    for (int i = 0; i < M; ++i) {
        X[i] = (2 * i - M + 1) * d / 2.0;
    }
    return X;
}

// Gaussian function
std::complex<double> G(std::complex<double> z) {
    return std::exp(-std::norm(z)) / M_PI;
}

int main() {
    // Define constellation sizes to test
    const std::vector<int> M_values = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    std::vector<long long> computation_times(M_values.size());

    // Quadrature parameters
    const int N = 2;
    std::vector<double> nodes, weights;
    GaussHermite_Locations_Weights(N, nodes, weights);

    const double SNR = 1.0;

    // Loop over different constellation sizes
    for (size_t i = 0; i < M_values.size(); i++) {
        int M = M_values[i];
        double d = 2.0;

        // Generate and normalize constellation
        std::vector<double> X = generatePAMConstellation(M, d);
        double avg_energy = std::reduce(std::execution::par, X.begin(), X.end(), 0.0, [](double sum, double x) {
            return sum + x * x;
        }) / M;

        std::transform(std::execution::par, X.begin(), X.end(), X.begin(), [&](double x) {
            return x / std::sqrt(avg_energy);
        });

        // Equal probabilities
        std::vector<double> Q(M, 1.0 / M);

        // Rho values
        const std::vector<double> rho_values = {0.0}; // Example: Only one rho value for now
        std::vector<double> Eo_rho(rho_values.size());

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Loop over rho values
        for (size_t j = 0; j < rho_values.size(); j++) {
            double rho = rho_values[j];
            double integral_sum = 0.0;

            // Use parallel loop for the outer integral
            integral_sum = std::transform_reduce(
                    std::execution::par,
                    X.begin(),
                    X.end(),
                    0.0,
                    std::plus<>(),
                    [&](double x) {
                        double quadrature_sum = 0.0;

                        for (int k = 0; k < N; k++) {
                            for (int l = 0; l < N; l++) {
                                std::complex<double> z(nodes[k], nodes[l]);
                                double fp_sum = 0.0;

                                for (double x_bar : X) {
                                    std::complex<double> diff = z + std::sqrt(SNR) * (x - x_bar);
                                    fp_sum += Q[0] * std::pow(std::abs(G(diff)), 1.0 / (1.0 + rho));
                                }

                                fp_sum /= std::pow(std::abs(G(z)), 1.0 / (1.0 + rho));
                                quadrature_sum += weights[k] * weights[l] * std::pow(fp_sum, rho);
                            }
                        }
                        return Q[0] * quadrature_sum;
                    });

            Eo_rho[j] = -std::log2((1.0 / M_PI) * integral_sum);
        }

        // Stop timing and convert to milliseconds
        auto end = std::chrono::high_resolution_clock::now();
        computation_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Computation time for M = " << M << ": "
                  << computation_times[i] << " milliseconds\n";
    }

    // Display results table
    std::cout << "\nComputation Times for Different Constellation Sizes:\n";
    std::cout << std::setw(20) << "Constellation Size" << std::setw(20) << "Time (ms)\n";
    std::cout << std::string(40, '-') << "\n";

    for (size_t i = 0; i < M_values.size(); i++) {
        std::cout << std::setw(20) << M_values[i] << std::setw(20) << computation_times[i] << "\n";
    }

    return 0;
}
