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
    for (auto& d : in_data) {
        d -= mean;
    }
}

void ComputeFft(vector<double>& in_data, vector<complex<double>>& fdata) {
    int N = (int)fdata.size();
    //double *in = (double *) malloc(N * sizeof(double));
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

void ComputeCumulant(vector<complex<double>>& fdata, vector<vector<complex<double>>>& cmat) {
    double div = 1.0 / fdata.size();
    for (int k1 = 0, N = fdata.size(); k1 < N; ++k1) {
        for (int k2 = 0; k2 <= k1 && 2 * (k1 + k2) < N; ++k2) {
            cmat[k1][k2] = (fdata[k1] * fdata[k2] * conj(fdata[k1 + k2])) * div;
        }
    }
}

void Smooth(vector<vector<complex<double>>>& cmat, int M3, vector<vector<complex<double>>>& scmat) {
    int N = cmat.size(), n = N - M3 + 1;
    double M3sq = 1.0 / (M3 * M3);
    scmat.resize(n, vector<complex<double>>(n));
#pragma omp parallel for collapse(2) num_threads(total_thread)
    for (int k1 = 0; k1 < n; ++k1) {
        for (int k2 = 0; k2 < n; ++k2) {
            complex<double> w_sum = .0;
            for (int l1 = 0; l1 < M3; ++l1) {
                for (int l2 = 0; l2 < M3; ++l2) {
                    w_sum += cmat[k1 + l1][k2 + l2];
                }
            }
            scmat[k1][k2] = M3sq * w_sum;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "\nusage: bispec_naive_omp in_file_name\n" << endl;
        return EXIT_FAILURE;
    }

    total_thread = stoi(argv[2]);
    if (total_thread <= 0 || total_thread > omp_get_max_threads()) {
        total_thread = omp_get_max_threads();
    }

    TimePoint tp1 = Clock::now();
    string in_file_name(argv[1]);
    vector<double> in_data;
    ReadData(in_file_name, in_data); // read input time series data
    Reformat(in_data);

    int N = in_data.size();
    vector<complex<double>> fdata(N);
    ComputeFft(in_data, fdata); // compute fft of the series

    vector<vector<complex<double>>> cmat(N, vector<complex<double>>(N, (0, 0)));
    ComputeCumulant(fdata, cmat); // compute third order cumulant

    int M3 = (int)pow(1.0 * N, .625);
    vector<vector<complex<double>>> scmat;
    Smooth(cmat, M3, scmat);
    TimePoint tp2 = Clock::now();
    cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(tp2 - tp1).count() << "ms" << endl;
#ifdef DEBUG_MODE
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max memory used: " << usage.ru_maxrss << endl;
#endif
#ifdef DEBUG_WRITE
    WriteDebugDataBispec(scmat, "debug_data/mat_bispec_naive_omp.txt");
#endif

#ifdef DEBUG_MODE
    //cout << scmat.size() << " : " << M3 << endl;
#endif

    return EXIT_SUCCESS;
}