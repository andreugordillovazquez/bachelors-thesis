#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <complex>
#include <numeric>

using namespace std;

typedef complex<double> Complex;

// Gaussian function
inline double G(const Complex &z) {
    return (1.0 / M_PI) * exp(-norm(z));
}

// Generate PAM constellation (function stub)
vector<double> generatePAMConstellation(int M, double d) {
    vector<double> X(M);
    for (int i = 0; i < M; ++i) {
        X[i] = d * (2 * i - M + 1);
    }
    return X;
}

// Gauss-Hermite nodes and weights (function stub)
pair<vector<double>, vector<double>> GaussHermite_Locations_Weights(int N) {
    vector<double> nodes = {-0.707107, 0.707107};
    vector<double> weights = {0.886227, 0.886227};
    return make_pair(nodes, weights);
}

int main() {
    // Define constellation sizes to test
    vector<int> Array_M = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    // Initialize computation times and Eo_rho values
    vector<int> computation_times(Array_M.size());
    vector<double> Eo_final_values(Array_M.size());

    // Quadrature nodes and weights
    int N = 2;
    auto [nodes, weights] = GaussHermite_Locations_Weights(N);

    double SNR = 1.0;

    // Rho values
    double rho = 0.5;

    // Loop over different constellation sizes
    for (size_t i = 0; i < Array_M.size(); ++i) {
        int M = Array_M[i];
        double d = 2.0;

        // Generate constellation
        vector<double> X = generatePAMConstellation(M, d);

        // Normalize constellation to have unit energy
        double avg_energy = accumulate(X.begin(), X.end(), 0.0, [](double sum, double x) { return sum + x * x; }) / M;
        for (double &x : X) {
            x /= sqrt(avg_energy);
        }

        // Equal probabilities
        vector<double> Q(M, 1.0 / M);

        double Eo_rho = 0.0;

        // Start the timer
        auto start = chrono::high_resolution_clock::now();

        double integral_sum = 0.0;

        // Loop over PAM symbols
        for (size_t x_idx = 0; x_idx < X.size(); ++x_idx) {
            double x = X[x_idx];
            double quadrature_sum = 0.0;

            // Nested loop over quadrature nodes
            for (size_t k = 0; k < nodes.size(); ++k) {
                for (size_t l = 0; l < nodes.size(); ++l) {
                    Complex z(nodes[k], nodes[l]);
                    double fp_sum = 0.0;

                    for (size_t x_bar_idx = 0; x_bar_idx < X.size(); ++x_bar_idx) {
                        double x_bar = X[x_bar_idx];
                        double term1 = log(G(z + sqrt(SNR) * x - sqrt(SNR) * x_bar));
                        fp_sum += Q[x_idx] * exp((1.0 / (1.0 + rho)) * term1);
                    }

                    double term2 = log(G(z));
                    double fp = fp_sum / exp((1.0 / (1.0 + rho)) * term2);
                    double term3 = log(fp);
                    quadrature_sum += weights[k] * weights[l] * exp(rho * term3);
                }
            }

            integral_sum += Q[x_idx] * quadrature_sum;
        }

        double term4 = log((1.0 / M_PI) * integral_sum);
        Eo_rho = -log2(exp(term4));

        // Record computation time
        auto end = chrono::high_resolution_clock::now();
        computation_times[i] = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        Eo_final_values[i] = Eo_rho;

        // Display computation time
        cout << "Computation time for M = " << M << ": " << computation_times[i] << " milliseconds\n";
    }

    // Display results
    cout << "Computation Times for Different Constellation Sizes:\n";
    for (size_t i = 0; i < Array_M.size(); ++i) {
        cout << "M = " << Array_M[i] << ", Time = " << computation_times[i] << " ms\n";
    }

    return 0;
}
