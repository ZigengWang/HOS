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
    in_data.erase(in_data.begin());
}

void Reformat(vector<double>& in_data) {
    //cout << "Reformat" << endl;
    int N = in_data.size();
    double sm = accumulate(in_data.begin(), in_data.end(), .0);
    double mean = sm / N;
#pragma omp parallel for num_threads(total_thread)
    for (int i = 0; i < N; ++i) {
        in_data[i] -= mean;
    }
}

void ComputeFft(vector<double>& in_data, vector<complex<double>>& fdata) {
    //cout << "FFT" << endl;
    int N = (int)fdata.size();
    fftw_init_threads();
    fftw_complex *in = (fftw_complex *) fftw_malloc(N * sizeof(fftw_complex));
    fftw_complex *out = (fftw_complex *) fftw_malloc(N * sizeof(fftw_complex));
#pragma omp parallel for num_threads(total_thread)
    for (int i = 0; i < N; ++i) {
        in[i][0] = in_data[i];
        in[i][1] = 0;
    }
    fftw_plan_with_nthreads(total_thread);
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
#pragma omp parallel for num_threads(total_thread)
    for (int i = 0; i < N; ++i) {
        fdata[i] = {out[i][REAL], out[i][IMAG]};
    }
    fftw_destroy_plan(p);
    fftw_cleanup_threads();
    fftw_free(out);
    fftw_free(in);
}

void ComputeCumulant(vector<complex<double>>& fdata, vector<vector<complex<double>>>& cmat) {
    double div = 1.0 / fdata.size();
    int N = fdata.size();
    //cout << "CUMULant" << endl;
#pragma omp parallel for num_threads(total_thread)
    for (int k1 = 0; k1 < N; ++k1) {
        for (int k2 = 0; k2 <= k1 && 2 * (k1 + k2) < N; ++k2) {
            cmat[k1][k2] = (fdata[k1] * fdata[k2] * conj(fdata[k1 + k2])) * div;
        }
    }
}

void Smooth(vector<complex<double>>& fdata, int M3, vector<vector<vector<complex<double>>>>& scmat) {
    double div = 1.0 / fdata.size();
    int N = fdata.size(), n = N - M3 + 1;
    double M3sq = 1.0 / (M3 * M3);
    int row_per_thread = (n + total_thread - 1) / total_thread, ex_row_per_thread = row_per_thread + M3 - 1;
    //cout << row_per_thread << " " << total_thread << " " << ex_row_per_thread << " " << M3 << endl;
#pragma omp parallel num_threads(total_thread)
    {
        vector<vector<complex<double>>> ccmat(ex_row_per_thread, vector<complex<double>>(N, (0, 0)));
        int thread_id = omp_get_thread_num(), s_ind = thread_id * row_per_thread, e_ind = min(s_ind + ex_row_per_thread, N);
        for (int i = s_ind, k = 0; i < e_ind; ++i, ++k) {
            for (int j = 0; j <= i && 2 * (i + j) < N; ++j) {
                ccmat[k][j] = (fdata[i] * fdata[j] * conj(fdata[i + j])) * div;
            }
        }
        for (int i = 0, l = e_ind - s_ind; i < l; ++i) {
            for (int j = 0; j < N; ++j) {
                ccmat[i][j] += (i? ccmat[i - 1][j] : 0) + (j? ccmat[i][j - 1] : 0)
                                          - (i && j? ccmat[i - 1][j - 1] : 0);
            }
        }
        int req = e_ind - s_ind - M3 + 1;
        scmat[thread_id].resize(req, vector<complex<double>>(n, (0, 0)));
        for (int r = M3 - 1, k = 0; k < req; ++r, ++k) {
            for (int k2 = 0, c = M3 - 1; k2 < n; ++k2, ++c) {
                scmat[thread_id][k][k2] = M3sq * (ccmat[r][c]
                                        - (r >= M3 ? ccmat[r - M3][c] : 0)
                                        - (c >= M3 ? ccmat[r][c - M3] : 0)
                                        + (r >= M3 && c >= M3? ccmat[r - M3][c - M3] : 0));
            }
        }
    }
}

void SmoothRe(vector<vector<complex<double>>>& cmat, int M3, vector<vector<complex<double>>>& scmat) {
    int N = cmat.size(), n = N - M3 + 1;
    double M3sq = 1.0 / (M3 * M3);
    scmat.resize(n, vector<complex<double>>(n));
    int row_per_thread = (N - M3 + 1 + total_thread - 1) / total_thread, ex_row_per_thread = row_per_thread + M3 - 1;
    vector<vector<vector<complex<double>>>> ccmat(total_thread, vector<vector<complex<double>>>(ex_row_per_thread,
                                                                                                vector<complex<double>>(N, (0, 0))));
#pragma omp parallel num_threads(total_thread)
    {
        int thread_id = omp_get_thread_num(), s_ind = thread_id * row_per_thread, e_ind = min(s_ind + ex_row_per_thread, N);
        //cout << thread_id << " " << s_ind << " " << e_ind << " " << M3 << " " << row_per_thread << endl;
        for (int i = s_ind, k = 0; i < e_ind; ++i, ++k) {
            for (int j = 0; j < N; ++j) {
                ccmat[thread_id][k][j] = cmat[i][j];
            }
        }
        for (int i = 0; i < ex_row_per_thread; ++i) {
            for (int j = 0; j < N; ++j) {
                ccmat[thread_id][i][j] += (i? ccmat[thread_id][i - 1][j] : 0) + (j? ccmat[thread_id][i][j - 1] : 0)
                                          - (i && j? ccmat[thread_id][i - 1][j - 1] : 0);
            }
        }
        for (int k1 = s_ind, r = M3 - 1; k1 < min(n, s_ind + row_per_thread); ++k1, ++r) {
            for (int k2 = 0, c = M3 - 1; k2 < n; ++k2, ++c) {
                scmat[k1][k2] = M3sq * (ccmat[thread_id][r][c]
                                        - (r >= M3 ? ccmat[thread_id][r - M3][c] : 0)
                                        - (c >= M3 ? ccmat[thread_id][r][c - M3] : 0)
                                        + (r >= M3 && c >= M3? ccmat[thread_id][r - M3][c - M3] : 0));
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "\nusage: bispec_omp in_file_name num_threads\n" << endl;
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
    TimePoint tpa = Clock::now();
    cout << "time: " << chrono::duration_cast<chrono::milliseconds>(tpa - tp1).count() << "ms" << "\n";

    int N = in_data.size();
    vector<complex<double>> fdata(N);
    ComputeFft(in_data, fdata); // compute fft of the series
    TimePoint tpb = Clock::now();
    cout << "time: " << chrono::duration_cast<chrono::milliseconds>(tpb - tpa).count() << "ms" << "\n";

    int M3 = (int)pow(1.0 * N, .625);
    vector<vector<vector<complex<double>>>> scmat(total_thread);
    Smooth(fdata, M3, scmat);
    TimePoint tp2 = Clock::now();
    cout << "time: " << chrono::duration_cast<chrono::milliseconds>(tp2 - tpb).count() << "ms" << "\n";
    cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(tp2 - tp1).count() << "ms" << endl;
#ifdef DEBUG_MODE
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max memory used: " << usage.ru_maxrss << endl;
#endif
#ifdef DEBUG_WRITE
    WriteDebugDataBispecOMP(scmat, "debug_data/mat_bispec_omp.txt");
#endif

#ifdef DEBUG_MODE
    //cout << scmat.size() << " : " << M3 << endl;
#endif

    return 0;
}