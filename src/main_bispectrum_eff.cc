/*
 * Time: O(N^2) where N is series length
 * Space: O(M^2) where M X M is window size
 */
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
    int N = fdata.size(), MM3 = M3 + 1, n = N - M3 + 1;
    double div = 1.0 / N, M3sq = 1.0 / (M3 * M3);
#ifdef DEBUG_WRITE
    vector<vector<complex<double>>> d_out(n, vector<complex<double>>(n));
#endif
    for (int i = 0, M = M3 + M3 - 1; i < N; i += M3) {
        vector<complex<double>> out(M3 * M3);
        vector<vector<complex<double>>> scmat(M, vector<complex<double>>(M, .0));
        for (int j = 0; j < N; j += M3) {
            int k1m = i + M >= N? N - i : M;
            int k2m = j + M >= N? N - j : M;
            for (int k1 = 0, ind = 0; k1 < k1m; ++k1) {
                for (int k2 = 0; k2 < k2m; ++k2) {
                    int r = i + k1, c = j + k2;
                    scmat[k1][k2] = c <= r && 2 * (c + r) < N ? (fdata[c] * fdata[r] * conj(fdata[c + r])) * div : .0;
                    scmat[k1][k2] = scmat[k1][k2]
                                    + (k1? scmat[k1 - 1][k2] : 0)
                                    + (k2? scmat[k1][k2 - 1] : 0)
                                    - (k1 && k2? scmat[k1 - 1][k2 - 1] : 0);
                    if (k1 + 1 >= M3 && k2 + 1 >= M3) {
                        out[ind++] = M3sq * (scmat[k1][k2]
                                             - (k1 >= M3? scmat[k1 - M3][k2] : 0)
                                             - (k2 >= M3? scmat[k1][k2 - M3] : 0)
                                             + (k1 >= M3 && k2 >= M3? scmat[k1 - M3][k2 - M3] : 0));
#ifdef DEBUG_WRITE
                        d_out[r - M3 + 1][c - M3 + 1] = out[ind - 1];
#endif
                    }
                }
            }
        }
    }
#ifdef DEBUG_WRITE
    ofstream out_file("debug_data/mat_bispec_eff.txt");
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
        cout << "\nusage: bispec_eff in_file_name\n" << endl;
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

    return 0;
}
