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
    in_data.erase(in_data.begin()); // erase if first value is id
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
    int N = fdata.size();
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
    int N = fdata.size(), M = M3 + M3 - 1, n = N - M3 + 1;
    double div = 1.0 / N, M3cu = 1.0 / (M3 * M3 * M3);
#ifdef DEBUG_WRITE
    vector<vector<vector<complex<double>>>> d_out(n, vector<vector<complex<double>>>(n, vector<complex<double>>(n, 0)));
#endif
#pragma omp parallel for num_threads(total_thread)
    for (int i = 0; i < N; i += M3) {
        vector<complex<double>> out(M3 * M3 * M3);
        vector<vector<vector<complex<double>>>> scmat(M, vector<vector<complex<double>>>(M, vector<complex<double>>(M, .0)));
        for (int j = 0; j < N; j += M3) {
            for (int k = 0; k < N; k += M3) {
                int k1m = i + M >= N? N - i : M;
                int k2m = j + M >= N? N - j : M;
                int k3m = k + M >= N? N - k : M;
                int ind = 0;
                for (int k1 = 0; k1 < k1m; ++k1) {
                    for (int k2 = 0; k2 < k2m; ++k2) {
                        for (int k3 = 0; k3 < k3m; ++k3) {
                            int r1 = i + k1, r2 = j + k2, r3 = k + k3;
                            scmat[k1][k2][k3] = r2 <= r1 && r3 <= r2 && 2 * (r1 + r2 + r3) < N? fdata[r1] * fdata[r2] * fdata[r3] * conj(fdata[r1 + r2 + r3]) * div : .0;
                            scmat[k1][k2][k3] = scmat[k1][k2][k3] + (k1? scmat[k1 - 1][k2][k3] : .0) + (k2? scmat[k1][k2 - 1][k3] : .0) + (k3? scmat[k1][k2][k3 - 1] : .0)
                                                - (k1 && k2? scmat[k1 - 1][k2 - 1][k3] : .0) - (k1 && k3? scmat[k1 - 1][k2][k3 - 1] : .0) - (k2 && k3? scmat[k1][k2 - 1][k3 - 1] : .0)
                                                + (k1 && k2 && k3? scmat[k1 - 1][k2 - 1][k3 - 1] : .0);
                            if (k1 + 1 >= M3 && k2 + 1 >= M3 && k3 + 1 >= M3) {
                                out[ind++] = M3cu * (scmat[k1][k2][k3]
                                                     - (k1 >= M3? scmat[k1 - M3][k2][k3] : .0)
                                                     - (k2 >= M3? scmat[k1][k2 - M3][k3] : .0)
                                                     - (k3 >= M3? scmat[k1][k2][k3 - M3] : .0)
                                                     + (k1 >= M3 && k2 >= M3? scmat[k1 - M3][k2 - M3][k3] : .0)
                                                     + (k1 >= M3 && k3 >= M3? scmat[k1 - M3][k2][k3 - M3] : .0)
                                                     + (k2 >= M3 && k3 >= M3? scmat[k1][k2 - M3][k3 - M3] : .0)
                                                     - (k1 >= M3 && k2 >= M3 && k3 >= M3? scmat[k1 - M3][k2 - M3][k3 - M3] : .0));
#ifdef DEBUG_WRITE
                                d_out[r1 - M3 + 1][r2 - M3 + 1][r3 - M3 + 1] = out[ind - 1];
#endif
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef DEBUG_WRITE
    ofstream out_file("debug_data/mat_trispec_eff.txt");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (abs(d_out[i][j][k].real()) < threshold) d_out[i][j][k].real(0);
                if (abs(d_out[i][j][k].imag()) < threshold) d_out[i][j][k].imag(0);
                if (abs(d_out[i][j][k])) {
                    out_file << d_out[i][j][k] << "\n";
                }
            }
        }
    }
    out_file.close();
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "\nusage: trispec_eff_omp in_file_name num_threads\n" << endl;
        return EXIT_FAILURE;
    }

    total_thread = stoi(argv[2]);
    if (total_thread <= 0 || total_thread > omp_get_max_threads()) {
        total_thread = omp_get_max_threads();
    }

    string in_file_name(argv[1]);
    vector<double> in_data;
    ReadData(in_file_name, in_data); // read input time series data

    TimePoint tp1 = Clock::now();
    omp_set_num_threads(total_thread);
    Reformat(in_data); // subtract mean from each value

    int N = in_data.size();
    vector<complex<double>> fdata(N);
    ComputeFft(in_data, fdata); // compute fft of the series
    int M3 = (int)pow(1.0 * N, .625);
    ComputeCumulantSmooth(fdata, M3);

    TimePoint tp2 = Clock::now();
    cout << "Total threads: " << total_thread << " N: " << N << " M3: " << M3 << "\n";
    cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(tp2 - tp1).count() << "ms" << "\n";
#ifdef DEBUG_MODE
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max memory: " << usage.ru_maxrss << "KB" << "\n";
#endif
    return EXIT_SUCCESS;
}
