#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <complex>
#include <fftw3.h>
#include "omp.h"
#include "debug_hosa.h"
#include "config.h"
using namespace std;

#ifdef DEBUG_MODE
    #include <sys/resource.h>
#endif

enum TYPE {REAL, IMAG};

int total_thread;

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
#pragma omp parallel for num_threads(total_thread)
    for (int i = 0; i < in_data.size(); ++i) {
        in_data[i] -= mean;
    }
}

void ComputeFft(vector<double>& in_data, vector<complex<double>>& fdata) {
    int N = (int)fdata.size();
    fftw_complex *in = (fftw_complex *) fftw_malloc(N * sizeof(fftw_complex));
    fftw_complex *out = (fftw_complex *) fftw_malloc(N * sizeof(fftw_complex));
#pragma omp parallel for num_threads(total_thread)
    for (int i = 0; i < N; ++i) {
        in[i][0] = in_data[i];
        in[i][1] = 0;
    }
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
#pragma omp parallel for num_threads(total_thread)
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
    ofstream out_file("debug_data/mat_bispec_best_mem_omp.txt");
#endif

    const int lr = M3 - 1;
#pragma omp parallel for num_threads(total_thread)
    for (int k1 = 0; k1 < n; ++k1) {
        int ind = 0;
        vector<complex<double>> out(n);
        vector<vector<complex<double>>> scmat(M3, vector<complex<double>>(MM3));
        for (int i = 0; i < M3; ++i) {
            scmat[i][M3] = (0, 0);
        }
        for (int k2 = 0; k2 < N; ++k2) {
            int c = k2 % MM3, cm1 = (c - 1 + MM3) % MM3;
            for (int k3 = 0; k3 < M3; ++k3) {
                int rr = k1 + k3, r = k3;
                scmat[r][c] = k2 <= rr && 2 * (rr + k2) < N ? (fdata[rr] * fdata[k2] * conj(fdata[rr + k2])) * div
                                                            : (0, 0);
                scmat[r][c] = scmat[r][c]
                              + scmat[r][cm1]
                              + (r? scmat[r - 1][c] : 0)
                              - (r? scmat[r - 1][cm1] : 0);
            }
            if (k2 + 1 >= M3) {
                int cp1 = (c + 1) % MM3;
                out[ind++] = M3sq * (scmat[lr][c]
                                     - scmat[lr][cp1]);
            }
        }
#ifdef DEBUG_WRITE
        if (ind) {
            for (int i = 0; i < ind; ++i) {
                if (abs(out[i].real()) < threshold) out[i].real(0);
                if (abs(out[i].imag()) < threshold) out[i].imag(0);
                if (abs(out[i])) {
                    out_file << out[i] << "\n";
                }
            }
        }
#endif
    }
#ifdef DEBUG_WRITE
    out_file.close();
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "\nusage: bispec_best_mem_omp in_file_name num_threads\n" << endl;
        exit(1);
    }

    total_thread = stoi(argv[2]);
    if (total_thread <= 0 || total_thread > omp_get_max_threads()) {
        total_thread = omp_get_max_threads();
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