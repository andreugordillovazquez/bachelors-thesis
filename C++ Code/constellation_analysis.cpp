#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <iomanip>

// Helper function to generate Gauss-Hermite quadrature nodes and weights
void GaussHermite_Locations_Weights(int N, std::vector<double>& nodes, std::vector<double>& weights) {
    // For N = 2, hardcoded values (for demonstration)
    if (N == 2) {
        nodes = {-0.7071067811865475, 0.7071067811865475};
        weights = {0.8862269254527580, 0.8862269254527580};
    }
}

// Generate PAM constellation
std::vector<double> generatePAMConstellation(int M, double d) {
    std::vector<double> X(M);
    for (int i = 0; i < M; ++i) {
        X[i] = (2 * i - M + 1) * (d / 2);
    }
    return X;
}

// Gaussian function
inline double G_magnitude(const std::complex<double>& z) {
    return (1.0 / M_PI) * std::exp(-std::norm(z));
}

int main() {
    // Define constellation sizes to test
    std::vector<int> M_values = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
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

        // Generate constellation
        std::vector<double> X = generatePAMConstellation(M, d);

        // Normalize constellation
        double avg_energy = 0.0;
        for (double x : X) {
            avg_energy += x * x;
        }
        avg_energy /= X.size();

        for (double& x : X) {
            x /= std::sqrt(avg_energy);
        }

        // Equal probabilities
        std::vector<double> Q(M, 1.0 / M);

        // Rho values
        double rho = 0.5; // Fixed rho value for simplicity
        double Eo_rho = 0.0;

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        double integral_sum = 0.0;
        for (size_t x_idx = 0; x_idx < X.size(); x_idx++) {
            double x = X[x_idx];
            double quadrature_sum = 0.0;

            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    std::complex<double> z(nodes[k], nodes[l]);
                    double fp_sum = 0.0;

                    for (size_t x_bar_idx = 0; x_bar_idx < X.size(); x_bar_idx++) {
                        double x_bar = X[x_bar_idx];
                        std::complex<double> diff = z + std::sqrt(SNR) * (x - x_bar);
                        fp_sum += Q[x_idx] * std::exp((1.0 / (1.0 + rho)) * std::log(G_magnitude(diff)));
                    }

                    fp_sum /= std::exp((1.0 / (1.0 + rho)) * std::log(G_magnitude(z)));
                    quadrature_sum += weights[k] * weights[l] * std::exp(rho * std::log(fp_sum));
                }
            }
            integral_sum += Q[x_idx] * quadrature_sum;
        }

        Eo_rho = -std::log2(std::exp(std::log((1.0 / M_PI) * integral_sum)));

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