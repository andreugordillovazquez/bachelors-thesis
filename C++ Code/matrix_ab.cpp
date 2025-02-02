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

// Main function
int main() {
    // Initialize parameters
    std::vector<int> Array_M = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1028};
    std::vector<double> computation_times(Array_M.size());
    double SNR = 1.0;
    std::vector<double> rho_values = {0.5};  // Changed to 1.0 for E0(1)
    
    // Get Gauss-Hermite nodes and weights for N = 2
    auto nodes = GaussHermite_Locations_Weights_Nodes();
    auto weights = GaussHermite_Locations_Weights_Weights();
    
    // Define Gaussian PDF function
    auto G = [](std::complex<double> z) {
        return (1.0 / M_PI) * exp(-std::norm(z));
    };
    
    std::cout << std::fixed << std::setprecision(6);  // Increased precision for E0 values
    std::cout << "Constellation Size (M) | E0(1) | Computation Time (ms)\n";
    std::cout << "------------------------------------------------\n";
    
    for(size_t i = 0; i < Array_M.size(); i++) {
        int M = Array_M[i];
        double d = 2.0;
        
        // Generate and normalize constellation
        auto X = generatePAMConstellation(M, d);
        
        // Calculate average energy
        double avg_energy = 0.0;
        for(double x : X) {
            avg_energy += x * x;
        }
        avg_energy /= X.size();
        
        // Normalize constellation
        for(double& x : X) {
            x /= sqrt(avg_energy);
        }
        
        // Equal probabilities
        std::vector<double> Q(M, 1.0 / M);
        
        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();
        
        auto z_matrix = createComplexNodesMatrix(nodes);
        auto pi_matrix = createPiMatrix(X.size(), nodes.size(), weights);
        auto g_matrix = createGMatrix(X, z_matrix, SNR, G);
        
        std::vector<double> Eo_rho(rho_values.size());
        for(size_t idx = 0; idx < rho_values.size(); idx++) {
            double rho = rho_values[idx];
            Eo_rho[idx] = computeEoForRho(rho, Q, pi_matrix, g_matrix);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        computation_times[i] = duration.count();
        
        std::cout << std::setw(20) << M << " | "
                  << std::setw(12) << Eo_rho[0] << " | "  // Display E0(1)
                  << std::setw(20) << computation_times[i] << "\n";
    }
    
    return 0;
}

// Function implementations
std::vector<double> GaussHermite_Locations_Weights_Nodes() {
    // These are the nodes for N = 2
    return {-0.7071067811865475, 0.7071067811865475};
}

std::vector<double> GaussHermite_Locations_Weights_Weights() {
    // These are the weights for N = 2
    return {0.8862269254527580, 0.8862269254527580};
}

std::vector<double> generatePAMConstellation(int M, double d) {
    std::vector<double> pam;
    pam.reserve(M);
    
    double start = -((M-1.0)/2.0);
    double end = ((M-1.0)/2.0);
    
    for(double i = start; i <= end + 0.1; i++) {  // +0.1 for floating point precision
        pam.push_back(i * d);
    }
    
    return pam;
}

std::vector<std::vector<std::complex<double>>> createComplexNodesMatrix(const std::vector<double>& nodes) {
    int N = nodes.size();
    std::vector<std::vector<std::complex<double>>> z_matrix(N, std::vector<std::complex<double>>(N));
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            z_matrix[i][j] = std::complex<double>(nodes[i], nodes[j]);
        }
    }
    
    return z_matrix;
}

std::vector<std::vector<double>> createPiMatrix(int M, int N, const std::vector<double>& weights) {
    std::vector<std::vector<double>> pi_matrix(M, std::vector<double>(N * N * M, 0.0));
    
    for(int i = 0; i < M; i++) {
        int idx = 0;
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                // Calculate weight product
                double pi_block = weights[j] * weights[k];
                // Calculate column position
                int col = i * N * N + idx;
                // Assign the weight product
                pi_matrix[i][col] = pi_block;
                idx++;
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
    std::vector<std::vector<double>> g_matrix(M, std::vector<double>(N * N * M, 0.0));
    
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            for(int k = 0; k < N; k++) {
                for(int l = 0; l < N; l++) {
                    // Compute column index
                    int col_idx = j * N * N + k * N + l;
                    
                    // Compute G values
                    std::complex<double> arg = z_matrix[k][l] + 
                        (std::sqrt(SNR) * X[i]) - 
                        (std::sqrt(SNR) * X[j]);
                    
                    g_matrix[i][col_idx] = G(arg);
                }
            }
        }
    }
    
    return g_matrix;
}

double computeEoForRho(double rho, 
    const std::vector<double>& Q, 
    const std::vector<std::vector<double>>& pi_matrix, 
    const std::vector<std::vector<double>>& g_matrix) {
    
    int M = Q.size();
    int N2M = pi_matrix[0].size();
    
    // Compute g_powered
    std::vector<std::vector<double>> g_powered(M, std::vector<double>(N2M));
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N2M; j++) {
            g_powered[i][j] = pow(g_matrix[i][j], 1.0 / (1.0 + rho));
        }
    }
    
    // Compute qg_rho
    std::vector<double> qg_rho(N2M, 0.0);
    for(int j = 0; j < N2M; j++) {
        for(int i = 0; i < M; i++) {
            qg_rho[j] += Q[i] * g_powered[i][j];
        }
    }
    
    // Compute qg_rho_t
    std::vector<double> qg_rho_t(N2M);
    for(int j = 0; j < N2M; j++) {
        qg_rho_t[j] = pow(qg_rho[j], rho);
    }
    
    // Compute pi_g_rho and final component
    double component_second = 0.0;
    for(int i = 0; i < M; i++) {
        double sum = 0.0;
        for(int j = 0; j < N2M; j++) {
            double pi_g_rho = pi_matrix[i][j] * pow(g_matrix[i][j], -rho / (1.0 + rho));
            sum += pi_g_rho * qg_rho_t[j];
        }
        component_second += Q[i] * sum;
    }
    
    return -log2((1.0 / M_PI) * component_second);
}