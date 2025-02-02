#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <functional>
#include <complex>

// Function declarations
std::vector<double> GaussHermite_Locations_Weights_Nodes();
std::vector<double> GaussHermite_Locations_Weights_Weights();
std::vector<double> generatePAMConstellation(int M, double d);
std::vector<std::vector<std::complex<double>>> createComplexNodesMatrix(const std::vector<double>& nodes);
std::vector<std::vector<double>> createPiMatrix(int M, int N, const std::vector<double>& weights);
std::vector<std::vector<double>> createGMatrix(const std::vector<double>& X, 
    const std::vector<std::vector<std::complex<double>>>& z_matrix, 
    double SNR, 
    const std::function<double(std::complex<double>)>& G);
double computeEoForRho(double rho, 
    const std::vector<double>& Q, 
    const std::vector<std::vector<double>>& pi_matrix, 
    const std::vector<std::vector<double>>& g_matrix);

int main() {
    // Initialize parameters
    std::vector<int> Array_M = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1028};
    std::vector<double> computation_times(Array_M.size());
    double SNR = 1.0;
    std::vector<double> rho_values = {1.0};  // E0(1)

    // Get Gauss-Hermite nodes and weights for N = 2
    auto nodes = GaussHermite_Locations_Weights_Nodes();
    auto weights = GaussHermite_Locations_Weights_Weights();

    // Define Gaussian PDF function
    auto G = [](std::complex<double> z) {
        return (1.0 / M_PI) * exp(-std::norm(z));
    };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Constellation Size (M) | E0(1) | Computation Time (ms)\n";
    std::cout << "------------------------------------------------\n";

    for (size_t i = 0; i < Array_M.size(); i++) {
        int M = Array_M[i];
        double d = 2.0;

        // Generate and normalize constellation
        auto X = generatePAMConstellation(M, d);

        // Calculate and normalize average energy
        double avg_energy = 0.0;
        for (double x : X) avg_energy += x * x;
        avg_energy = sqrt(avg_energy / X.size());

        for (double& x : X) x /= avg_energy;

        // Equal probabilities
        std::vector<double> Q(M, 1.0 / M);

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        auto z_matrix = createComplexNodesMatrix(nodes);
        auto pi_matrix = createPiMatrix(M, nodes.size(), weights);
        auto g_matrix = createGMatrix(X, z_matrix, SNR, G);

        std::vector<double> Eo_rho;
        Eo_rho.reserve(rho_values.size());
        for (double rho : rho_values) {
            Eo_rho.push_back(computeEoForRho(rho, Q, pi_matrix, g_matrix));
        }

        auto end = std::chrono::high_resolution_clock::now();
        computation_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << std::setw(20) << M << " | "
                  << std::setw(12) << Eo_rho[0] << " | "
                  << std::setw(20) << computation_times[i] << "\n";
    }

    return 0;
}

// Function implementations
std::vector<double> GaussHermite_Locations_Weights_Nodes() {
    return {-0.7071067811865475, 0.7071067811865475};
}

std::vector<double> GaussHermite_Locations_Weights_Weights() {
    return {0.8862269254527580, 0.8862269254527580};
}

std::vector<double> generatePAMConstellation(int M, double d) {
    std::vector<double> pam;
    pam.reserve(M);

    double start = -((M - 1) / 2.0);
    for (int i = 0; i < M; ++i) {
        pam.push_back((start + i) * d);
    }

    return pam;
}

std::vector<std::vector<std::complex<double>>> createComplexNodesMatrix(const std::vector<double>& nodes) {
    int N = nodes.size();
    std::vector<std::vector<std::complex<double>>> z_matrix(N, std::vector<std::complex<double>>(N));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            z_matrix[i][j] = {nodes[i], nodes[j]};
        }
    }

    return z_matrix;
}

std::vector<std::vector<double>> createPiMatrix(int M, int N, const std::vector<double>& weights) {
    std::vector<std::vector<double>> pi_matrix(M, std::vector<double>(N * N * M));

    for (int i = 0; i < M; i++) {
        for (int j = 0, idx = 0; j < N; j++) {
            for (int k = 0; k < N; k++, idx++) {
                pi_matrix[i][i * N * N + idx] = weights[j] * weights[k];
            }
        }
    }

    return pi_matrix;
}

std::vector<std::vector<double>> createGMatrix(
    const std::vector<double>& X, 
    const std::vector<std::vector<std::complex<double>>>& z_matrix, 
    double SNR, 
    const std::function<double(std::complex<double>)>& G) {
    
    int M = X.size();
    int N = z_matrix.size();
    double sqrtSNR = sqrt(SNR);

    std::vector<std::vector<double>> g_matrix(M, std::vector<double>(N * N * M));

    for (int i = 0; i < M; i++) {
        double sqrtSNR_Xi = sqrtSNR * X[i];
        for (int j = 0; j < M; j++) {
            double sqrtSNR_Xj = sqrtSNR * X[j];
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    g_matrix[i][j * N * N + k * N + l] = G(z_matrix[k][l] + sqrtSNR_Xi - sqrtSNR_Xj);
                }
            }
        }
    }

    return g_matrix;
}

// Missing Function Added: computeEoForRho
double computeEoForRho(double rho, 
    const std::vector<double>& Q, 
    const std::vector<std::vector<double>>& pi_matrix, 
    const std::vector<std::vector<double>>& g_matrix) {

    int M = Q.size();
    int N2M = pi_matrix[0].size();

    std::vector<std::vector<double>> g_powered(M, std::vector<double>(N2M));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N2M; j++) {
            g_powered[i][j] = exp((1.0 / (1.0 + rho)) * log(g_matrix[i][j]));
        }
    }

    std::vector<double> qg_rho(N2M, 0.0);
    for (int j = 0; j < N2M; j++) {
        for (int i = 0; i < M; i++) {
            qg_rho[j] += Q[i] * g_powered[i][j];
        }
    }

    std::vector<double> qg_rho_t(N2M);
    for (int j = 0; j < N2M; j++) {
        qg_rho_t[j] = exp(rho * log(qg_rho[j]));
    }

    std::vector<std::vector<double>> pi_g_rho(M, std::vector<double>(N2M));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N2M; j++) {
            pi_g_rho[i][j] = pi_matrix[i][j] * exp((-rho / (1.0 + rho)) * log(g_matrix[i][j]));
        }
    }

    double component_second = 0.0;
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = 0; j < N2M; j++) {
            sum += pi_g_rho[i][j] * qg_rho_t[j];
        }
        component_second += Q[i] * sum;
    }

    return -log2((1.0 / M_PI) * component_second);
}