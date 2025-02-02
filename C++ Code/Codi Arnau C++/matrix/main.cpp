#include "functions.h"
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <string>
#include <algorithm>
#include <fstream>
#include <locale>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

double minRate=0, maxRate=1, ptsRate = 4 /*10*/, incrementRate;
double minSNR=1.0,  maxSNR=1.0, ptsSNR = 1, incrementSNR; // todo change
int minmodexp = 1;
int maxmodexp = 10;
double minMod=pow(2,minmodexp),  maxMod=pow(2,maxmodexp), ptsMod=maxmodexp-minmodexp+1, incrementMod;
int indexRate = 1, indexSNR = 2, indexMod = 3, indexE0 = 4;
int constellation = 2;

bool mode1=false, mode2=false, mode3=false;

std::chrono::microseconds sum(vector<std::chrono::microseconds> vector1) {
    std:chrono::microseconds s = (std::chrono::microseconds) 0;
    for(auto x: vector1){
        s += x;
    }
    return s;
}

struct CustomNumpunct : std::numpunct<char> {
protected:
    char do_decimal_point() const override {
        return ','; // Use ',' as the decimal separator
    }
};

int main() {

    vector <vector<double>> e0_1_samples;
    vector <vector<double>> e0_2_samples;
    vector <vector<double>> e0_3_samples;
    vector<double> rates, snrs;
    vector<int> mods;

    // Generate rates, snrs, and mods based on the input ranges and pts
    if (ptsRate == 1) {
        rates.push_back(minRate);
    } else {
        for (int i = 0; i < ptsRate; ++i) {
            double rate = minRate + i * (maxRate - minRate) / (ptsRate - 1);
            rates.push_back(rate);
        }
    }

    if (ptsSNR == 1) {
        snrs.push_back(minSNR);
    } else {
        for (int i = 0; i < ptsSNR; ++i) {
            double snr = minSNR + i * (maxSNR - minSNR) / (ptsSNR - 1);
            snrs.push_back(snr);
        }
    }

    // Generate mods as powers of 2 within the specified range
    if (ptsMod == 1) {
        mods.push_back(minMod);
    }
    else {
        int currentMod = 1;
        while (currentMod <= maxMod) {
            if (currentMod >= minMod) {
                mods.push_back(currentMod);
            }
            currentMod *= 2;
        }
    }

    std::string constel;

    if (constellation == 1) {
        constel = "psk";
    } else if (constellation == 2) {
        constel = "pam";
    } else {
        constel = "secret"; //todo: buggy (produces nan) --> implement
    }

    // Gradient descent parameters
    int it = 50, n = 15; // only for GD/NAG!

    // --- SETTERS ---
    compute_hweights(n, it);
    setMod(mods[0], constel);

    auto start3 = std::chrono::high_resolution_clock::now();

    auto stop3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3);
    std::cout << endl << "time filling matrices: " << duration3.count() << " microseconds" << '\n';
    // --------------



    double e0_1, e0_2, e0_3;
    int totalIterations = rates.size() * snrs.size() * mods.size();
    int currentIteration = 0;
    auto deltas_sum = 0;
    double rho = 0.01, r = 1;
    unordered_map<int, vector<double>> alphadict;

    for (int k = 0; k < mods.size(); k++) {
        std::vector<double> alphas(mods[k], 0.01);
        alphadict[mods[k]] = alphas;
    }

    std::ofstream times_file("times_e0.txt");
    if (!times_file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    std::ofstream times_file_raw("times_e0_raw.txt");
    if (!times_file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    rates = {0.5};
    snrs = {1.0};
    //mods = {4};

    // Calculate e0_samples using gradient descent for each combination of rate, snr, and mod
    for (int i = 0; i < rates.size(); ++i) {
        for (int j = 0; j < snrs.size(); ++j) {
            vector<double> e0_row_1, e0_row_2, e0_row_3;

            for (int k = 0; k < mods.size(); ++k) {

                cout << "m: " << to_string(mods[k]) << endl;
                setMod(mods[k], constel);
                // matrix Q
                setQ();

                for (int n_ = 2; n_ <= 2; n_++) {
                    cout << "n_: " << n_ << endl;
                    auto start = std::chrono::high_resolution_clock::now();
                    vector<double> e0_samples;

                    //qDebug() << "i:" << i << " j:" << j << " k:" << k;
                    //qDebug() << "modval: " << mods[k];
                    setR(rates[i]);
                    setSNR(snrs[j]);

                    ///// CODE FOR THE ERROR TEST
                    // n
                    setN(n_);

                    // matrices
                    setPI(); setW();

                    vector<double> hweights = getAllHweights();
                    vector<double> roots = getAllRoots();
                    vector<double> multhweights = getAllMultHweights();

                    double grad_rho, grad_2_rho; // placeholders
                    double e0ptr, e0;
                    double incr = .01;

                    std::ofstream e0_file("out_m_" + to_string(mods[k]) + "_" + constel + "_n_" + to_string(n_) +".txt");
                    if (!e0_file.is_open()) {
                        std::cerr << "Error opening file!" << std::endl;
                        return 1;
                    }
                    e0_file.imbue(std::locale(e0_file.getloc(), new CustomNumpunct));

                    auto start_XX = std::chrono::high_resolution_clock::now();

                    /*
                    for (double rho_ = -1 + incr; rho_ < 2 + incr; rho_ += incr) {
                        e0 = E_0_co(0, rho_, grad_rho, grad_2_rho, e0ptr, n - 1, hweights, multhweights, roots);
                        e0_file << fixed << setprecision(15) << e0 << endl;
                    }
                    */

                    //e0 = E_0_1_co();
                    e0 = E_0_co(0, .5, grad_rho, grad_2_rho, e0ptr, n - 1, hweights, multhweights, roots);
                    e0_file << fixed << setprecision(15) << e0 << endl;

                    auto stop_XX = std::chrono::high_resolution_clock::now();
                    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);

                    times_file << "m: " << to_string(mods[k]) + " " + constel + " n_: " + to_string(n_) << " time: " << duration_XX.count() << " microseconds" << endl;
                    times_file_raw << duration_XX.count() << endl;

                    ////

                    // setMod(mods[k], constel); // todo
                    double beta = 1; //todo change -1 / e02(n);
                    // cout << "beta: " << beta << endl;

                    if (mode1) { // iid mode
                        //rho = 0; r = 0;
                        //e0_1 = GD_iid(r, rho, 1/beta, it, n);
                        e0_1 = NAG_iid(r, rho, /*1/beta*/1 / 0.644112, it, n, 2.6);
                        e0_row_1.push_back(e0_1);
                    }
                    if (mode2) { // cost constraint mode
                        //rho = 0; r = 0;
                        //e0_2 = GD_cc(r, rho, 1/beta, it, n);
                        e0_2 = NAG_iid(r, rho, /*1/beta*/1 / 0.644112, it, n, 2.6);
                        e0_row_2.push_back(e0_2);
                    }
                    if (mode3) { // constant composition mode
                        //vector<double> alphas(mods[k],0); rho = 0;
                        e0_3 = GD_ccomp(/*alphas*/alphadict[mods[k]], rho, 1 / beta, it, n);
                        //e0_3 = NAG(// alphas alphadict[mods[k]], rho, 1/beta, it, n, 2.6);
                        e0_row_3.push_back(e0_3);
                    }

                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    //qDebug() << "modulation " << mods[k] << "Duration: " << duration.count() * pow(10,-6) << "s";
                    deltas_sum += duration.count();
                    //cout << duration.count() << endl;
                    ++currentIteration;

                    float percentage = (static_cast<float>(currentIteration) / totalIterations) * 100;
                    //qDebug() << "progress: " << percentage;
                }
            }
            if (mode1)
                e0_1_samples.push_back(e0_row_1);
            if (mode2)
                e0_2_samples.push_back(e0_row_2);
            if (mode3)
                e0_3_samples.push_back(e0_row_3);
        }
    }
    if (mode1) {
        for (int i = 0; i < rates.size(); ++i) {
            for (int j = 0; j < snrs.size(); ++j) {
                for (int k = 0; k < mods.size(); ++k) {
                    cout << rates[i] << "," << snrs[j] << "," << mods[k] << ","
                         << e0_1_samples[i * snrs.size() + j][k] << "\n";
                }
            }
        }
    }
    if (mode2) {
        for (int i = 0; i < rates.size(); ++i) {
            for (int j = 0; j < snrs.size(); ++j) {
                for (int k = 0; k < mods.size(); ++k) {
                    cout << rates[i] << "," << snrs[j] << "," << mods[k] << ","
                         << e0_2_samples[i * snrs.size() + j][k] << "\n";
                }
            }
        }
    }
    if (mode3) {
        for (int i = 0; i < rates.size(); ++i) {
            for (int j = 0; j < snrs.size(); ++j) {
                for (int k = 0; k < mods.size(); ++k) {
                    cout << rates[i] << "," << snrs[j] << "," << mods[k] << ","
                         << e0_3_samples[i * snrs.size() + j][k] << "\n";
                }
            }
        }
    }

    cout << "avg time: " << deltas_sum / totalIterations << endl;
    cout << "total time: " << deltas_sum << endl;


    cout << endl << "---------------------------------" << endl;

    cout << "avg time: " << deltas_sum / totalIterations << endl;
    cout << "total time: " << deltas_sum << endl;
    unordered_map<string, vector<chrono::microseconds>> e0times = get_times();

    //const unordered_map<string, vector<chrono::microseconds>>& e0times) {

    // Convert the unordered_map to a vector of pairs (for sorting)
    vector<pair<string, vector<chrono::microseconds>>> sorted_entries(e0times.begin(), e0times.end());

    // Define a sorting comparator that compares the sum of vector elements
    auto comparator = [](const pair<string, vector<chrono::microseconds>> &a,
                         const pair<string, vector<chrono::microseconds>> &b) {
        auto sumVector = [](const vector<chrono::microseconds> &v) {
            chrono::microseconds sum(0);
            for (const auto &val: v) {
                sum += val;
            }
            return sum;
        };

        return size(a.second) > size(b.second); // Descending order
    };

    // Sort the vector of pairs using the comparator
    sort(sorted_entries.begin(), sorted_entries.end(), comparator);

    // Create a new unordered_map to store the sorted elements
    unordered_map<string, vector<chrono::microseconds>> sorted_map;

    // Insert sorted elements into the new unordered_map
    for (const auto &entry: sorted_entries) {
        sorted_map[entry.first] = entry.second;
    }


    for (auto pair_: sorted_entries) {
        if (!pair_.second.empty()) {
            cout << endl << pair_.first << ": " << size(pair_.second) << " times";
        }
    }
    cout << endl << "-------------" << endl;

    unordered_map<string, int> avgs; // average times

    for (auto pair_: e0times) {
        if (!pair_.second.empty()) {
            //cout << endl << pair_.first << " " << sum(pair_.second).count()/ size(pair_.second) << endl;
            avgs[pair_.first] = sum(pair_.second).count(); /// size(pair_.second);
        }
    }

    std::vector<std::pair<std::string, int>> sorted_avgs(avgs.begin(), avgs.end());

    std::sort(sorted_avgs.begin(), sorted_avgs.end(),
              [](const std::pair<std::string, int> &a, const std::pair<std::string, int> &b) {
                  return a.second > b.second;
              });

    for (const auto &pair: sorted_avgs) {
        std::cout << pair.first << ": " << pair.second << " microseconds" << '\n';
    }

    int sum = 0;

    for (const auto &pair: avgs) {
        sum += pair.second;
    }

    std::cout << "Sum of all times: " << sum << " microseconds" << '\n';

    vector<int> e0times2;
    /*
    for(double rho_ = 0; rho_ < 1; rho_ += 0.1){
        auto start2 = std::chrono::high_resolution_clock::now();
        E_0_co(rho_,0,0,n);
        auto stop2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
        e0times2.push_back(duration2.count());
        std::cout << "e0total: " << duration2.count() << " microseconds" << '\n';
    }
     */

    //cout << endl << "e0av: " << sum(e0times2)/e0times2.size() << endl;


    //E_0_co(0.5,0,0, n);

    // cout << "avg E0 time: "   << e0sum.count()/totalIterations << endl;
    // cout << "total E0 time: " << e0sum.count() << endl
}
