#ifndef __DEBUG_HOSA__

#define __DEBUG_HOSA__
#include <iostream>
#include <complex>
#include <iomanip>
#include "config.h"
using namespace std;

void WriteDebugDataBispec(vector<vector<complex<double>>>& data, string out_file_name) {
    ofstream out_file(out_file_name);
    out_file << fixed << setprecision(6);
    for (int N = data.size(), i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(data[i][j].real()) < threshold) data[i][j].real(0);
            if (abs(data[i][j].imag()) < threshold) data[i][j].imag(0);
            out_file << data[i][j] << " \n"[j + 1 == N];
        }
    }
    out_file.close();
}

void WriteDebugDataBispecFull(vector<vector<complex<double>>>& data, string out_file_name) {
    ofstream out_file(out_file_name);
    for (int n = data.size(), i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (abs(data[i][j].real()) < threshold) data[i][j].real(0);
            if (abs(data[i][j].imag()) < threshold) data[i][j].imag(0);
            //if (abs(data[i][j])) out_file << i << " " << j << ": " << data[i][j] << "\n";
            out_file << data[i][j] << "\t";
        }
        out_file << "\n";
    }
    out_file.close();
}

void WriteDebugDataTrispec(vector<vector<vector<complex<double>>>>& data, string out_file_name) {
    ofstream out_file(out_file_name);
    for (int n = data.size(), i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (abs(data[i][j][k].real()) < threshold) data[i][j][k].real(0);
                if (abs(data[i][j][k].imag()) < threshold) data[i][j][k].imag(0);
                //if (abs(data[i][j][k])) out_file << i << " " << j << " " << k << ": " << data[i][j][k] << "\n";
                if (abs(data[i][j][k])) {
                    out_file << data[i][j][k] << "\n";
                }
            }
        }
    }
    out_file.close();
}


void WriteDebugDataBispecOMP(vector<vector<vector<complex<double>>>>& data, string out_file_name) {
    ofstream out_file(out_file_name);
    for (int th = 0; th < data.size(); ++th) {
        int n = data[th].size(), N = data[th][0].size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < N; ++j) {
                if (abs(data[th][i][j].real()) < threshold) data[th][i][j].real(0);
                if (abs(data[th][i][j].imag()) < threshold) data[th][i][j].imag(0);
                //if (abs(data[i][j])) out_file << i << " " << j << ": " << data[i][j] << "\n";
                if (abs(data[th][i][j])) out_file << data[th][i][j] << "\n";
            }
        }
    }
    out_file.close();
}

#endif