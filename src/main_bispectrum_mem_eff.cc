#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <complex>
#include <fftw3.h>
#include "debug_hosa.h"
#include "config.h"
using namespace std;

#ifdef DEBUG_MODE
    #include <sys/resource.h>
#endif

enum TYPE {REAL, IMAG};

// read only the first series
void ReadData(string& in_file_name, vector<double>& in_data) {
    ifstream in_file(in_file_name);
    string series, val;
    in_file >> series;
    istringstream iss(series);
    while (getline(iss, val, ',')) {
        in_data.push_back(stod(val));
    }
    in_file.close();
    in_data.erase(in_data.begin());
}

void Reformat(vector<double>& in_data) {
    double sm = accumulate(in_data.begin(), in_data.end(), .0);
    double mean = sm / in_data.size();
    for (auto& d : in_data) {
        d -= mean;
    }
}

void ComputeFft(vector<double>& in_data, vector<complex<double>>& fdata) {
    int N = (int)fdata.size();
    fftw_complex *in = (fftw_complex *) fftw_malloc(N * sizeof(fftw_complex));
    fftw_complex *out = (fftw_complex *) fftw_malloc(N * sizeof(fftw_complex));
    for (int i = 0; i < N; ++i) {
        in[i][0] = in_data[i];
        in[i][1] = 0;
    }
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    for (int i = 0; i < N; ++i) {
        fdata[i] = {out[i][REAL], out[i][IMAG]};
    }
    fftw_destroy_plan(p);
    fftw_free(out);
    fftw_free(in);
}

void ComputeCumulantSmooth(vector<complex<double>>& fdata, int M3) {
    int N = fdata.size(), MM3 = M3 + 1, n = N - M3 + 1, M = M3 + M3 - 1;
    double div = 1.0 / N, M3sq = 1.0 / (M3 * M3);
#ifdef DEBUG_WRITE
    vector<vector<complex<double>>> d_out(n, vector<complex<double>>(n));
#endif
    for (int i = 0; i < N; i += M3) {
        vector<complex<double>> scmat(M, .0), tmp(M, .0);
        vector<complex<double>> premat(M, .0), pretmp(M, .0);
        for (int j = 0; j < N; ++j) {
            vector<complex<double>> out(M3);
            int m = i + M >= N? N - i : M, ind = 0;
            for (int k = 0; k < m; ++k) {
                int r = i + k, c = j;
                scmat[k] = c <= r && 2 * (c + r) < N ? (fdata[c] * fdata[r] * conj(fdata[c + r])) * div : .0;
                scmat[k] = scmat[k] + tmp[k] + (k? scmat[k - 1] : 0) - (k? tmp[k - 1] : 0);
                if (j + 1 >= M3 && k + 1 >= M3) {
                    if (j >= M3) {
                        int rr = i + k, cc = c - M3;
                        premat[k] = cc <= rr && 2 * (cc + rr) < N ? (fdata[cc] * fdata[rr] * conj(fdata[cc + rr])) * div : .0;
                        premat[k] = premat[k] + pretmp[k] + (k ? premat[k - 1] : 0) - (k ? pretmp[k - 1] : 0);
                        out[ind++] = M3sq * (scmat[k]
                                             - (k >= M3 ? scmat[k - M3] : 0)
                                             - premat[k]
                                             + (k >= M3? premat[k - M3] : 0));
                    } else {
                        out[ind++] = M3sq * (scmat[k] - (k >= M3? scmat[k - M3] : 0));
                    }
#ifdef DEBUG_WRITE
                    d_out[r - M3 + 1][c - M3 + 1] = out[ind - 1];
#endif
                }
            }
            //swap(premat, pretmp);
            //swap(scmat, tmp);
            for (int l = 0; l < M; ++l) {
                tmp[l] = scmat[l];
            }
            fill(scmat.begin(), scmat.end(), .0);
            for (int l = 0; l < M; ++l) {
                pretmp[l] = premat[l];
            }
            fill(premat.begin(), premat.end(), .0);
        }
    }
#ifdef DEBUG_WRITE
    ofstream out_file("debug_data/mat_bispec_mem_eff.txt");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (abs(d_out[i][j].real()) < threshold) d_out[i][j].real(0);
            if (abs(d_out[i][j].imag()) < threshold) d_out[i][j].imag(0);
            if (abs(d_out[i][j])) {
                out_file << d_out[i][j] << "\n";
            }
        }
    }
    out_file.close();
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "\nusage: bispec_mem_eff in_file_name\n" << endl;
        exit(1);
    }

    string in_file_name(argv[1]);
    vector<double> in_data;
    ReadData(in_file_name, in_data); // read input time series data

    TimePoint tp1 = Clock::now();
    Reformat(in_data);

    int N = in_data.size();
    vector<complex<double>> fdata(N);
    ComputeFft(in_data, fdata); // compute fft of the series

    int M3 = (int)pow(1.0 * N, .625);
    ComputeCumulantSmooth(fdata, M3); // compute third order cumulant

    TimePoint tp2 = Clock::now();
    cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(tp2 - tp1).count() << "ms" << endl;
#ifdef DEBUG_MODE
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max memory used: " << usage.ru_maxrss << endl;
#endif

#ifdef DEBUG_MODE
    //cout << scmat.size() << " : " << M3 << endl;
#endif

    return 0;
}