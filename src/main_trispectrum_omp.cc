#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <complex>
#include <fftw3.h>
#include <omp.h>
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
    int N = in_data.size();
    double sm = accumulate(in_data.begin(), in_data.end(), .0);
    double mean = sm / N;
#pragma omp parallel for num_threads(total_thread)
    for (int i = 0; i < N; ++i) {
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

void ComputeCumulant(vector<complex<double>>& fdata, vector<vector<vector<complex<double>>>>& cmat) {
    double div = 1.0 / fdata.size();
    int N = fdata.size();
#pragma omp parallel for num_threads(total_thread)
    for (int k1 = 0; k1 < N; ++k1) {
        for (int k2 = 0; k2 <= k1; ++k2) { // check this constraint
            for (int k3 = 0; k3 <= k2 && 2 * (k1 + k2 + k3) < N; ++k3) { // check this constraint
                cmat[k1][k2][k3] = (fdata[k1] * fdata[k2] * fdata[k3] * conj(fdata[k1 + k2 + k3])) * div;
            }
        }
    }
}

void Smooth(vector<vector<vector<complex<double>>>>& cmat, int M3,
            vector<vector<vector<complex<double>>>>& scmat) {
    int N = cmat.size(), n = N - M3 + 1;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                cmat[i][j][k] += (i? cmat[i - 1][j][k] : 0) + (j? cmat[i][j - 1][k] : 0) +
                        (k? cmat[i][j][k - 1] : 0) - (i && j? cmat[i - 1][j - 1][k] : 0) -
                        (j && k? cmat[i][j - 1][k - 1] : 0) - (i && k? cmat[i - 1][j][k - 1] : 0) +
                        (i && j && k? cmat[i - 1][j - 1][k - 1] : 0);
            }
        }
    }
    double M3cu = 1.0 / (M3 * M3 * M3);
    scmat.resize(n, vector<vector<complex<double>>>(n, vector<complex<double>>(n, {0, 0})));
#pragma omp parallel num_threads(total_thread)
    {
        int k1, r, k2, c, k3, d;
        for (k1 = 0, r = k1 + M3 - 1; k1 < n; ++k1, ++r) {
            for (k2 = 0, c = k2 + M3 - 1; k2 < n; ++k2, ++c) {
                for (k3 = 0, d = k3 + M3 - 1; k3 < n; ++k3, ++d) {
                    scmat[k1][k2][k3] = M3cu * (cmat[r][c][d]
                                                - (k1 ? cmat[k1 - 1][c][d] : 0)
                                                - (k2 ? cmat[r][k2 - 1][d] : 0)
                                                - (k3 ? cmat[r][c][k3 - 1] : 0)
                                                + (k1 && k2 ? cmat[k1 - 1][k2 - 1][d] : 0)
                                                + (k1 && k3 ? cmat[k1 - 1][c][k3 - 1] : 0)
                                                + (k2 && k3 ? cmat[r][k2 - 1][k3 - 1] : 0)
                                                - (k1 && k2 && k3 ? cmat[k1 - 1][k2 - 1][k3 - 1] : 0));
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "\nusage: trispec_omp in_file_name\n" << endl;
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
    Reformat(in_data); // subtract mean from each value

    int N = in_data.size();
    vector<complex<double>> fdata(N);
    ComputeFft(in_data, fdata); // compute fft of the series

    vector<vector<vector<complex<double>>>> cmat(N, vector<vector<complex<double>>>(N,
                                                 vector<complex<double>>(N, {0, 0})));
    ComputeCumulant(fdata, cmat); // compute fourth order cumulant

    int M3 = (int)pow(1.0 * N, .625);
    vector<vector<vector<complex<double>>>> scmat;
    Smooth(cmat, M3, scmat);

    TimePoint tp2 = Clock::now();
    cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(tp2 - tp1).count() << "ms" << "\n";
#ifdef DEBUG_MODE
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max memory: " << usage.ru_maxrss << "KB" << "\n";
#endif
#ifdef DEBUG_WRITE
    WriteDebugDataTrispec(scmat, "debug_data/mat_trispec_omp.txt");
#endif

#ifdef DEBUG_MODE
    //cout << scmat.size() << " : " << M3 << endl;
#endif

    return EXIT_SUCCESS;
}