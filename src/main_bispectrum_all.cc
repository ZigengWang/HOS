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

void ComputeCumulant(vector<complex<double>>& fdata, vector<vector<complex<double>>>& cmat) {
    double div = 1.0 / fdata.size();
    int N = fdata.size(), HN = N >> 1;
    for (int k1 = 0; k1 < N; ++k1) {
        for (int k2 = 0; k2 < N; ++k2) {
            cmat[(k1 + HN) % N][(k2 + HN) % N] = (fdata[k1] * fdata[k2] * conj(fdata[(k1 + k2) % N])) * div;
        }
    }
}

void Smooth(vector<vector<complex<double>>>& cmat, int N, int M3, vector<vector<complex<double>>>& scmat) {
    int HM3 = M3 >> 1, L = N + HM3;
    cout << N << " " << HM3 << " " << L << endl;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            cmat[i][j] += (i? cmat[i - 1][j] : 0) + (j? cmat[i][j - 1] : 0) - (i && j? cmat[i - 1][j - 1] : 0);
        }
    }
    double M3sq = 1.0 / (M3 * M3);
    scmat.resize(N, vector<complex<double>>(N, {0, 0}));
    for (int k1 = 0, r = k1 + HM3; k1 < N; ++k1, ++r) {
        for (int k2 = 0, c = k2 + HM3; k2 < N; ++k2, ++c) {
            scmat[k1][k2] = M3sq * (cmat[r][c] - (r >= M3? cmat[r - M3][c] : 0) -
                    (c >= M3? cmat[r][c - M3] : 0) + (r >= M3 && c >= M3? cmat[r - M3][c - M3] : 0));
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "\nusage: bispec in_file_name\n" << endl;
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
    M3 |= 1;
    int HM3 = M3 >> 1, L = N + HM3;
    cout << M3 << " X " << M3 << ": smoothing window size\n";

    vector<vector<complex<double>>> cmat(L, vector<complex<double>>(L, (0, 0)));
    ComputeCumulant(fdata, cmat); // compute third order cumulant

    vector<vector<complex<double>>> scmat;
    Smooth(cmat, N, M3, scmat);
    TimePoint tp2 = Clock::now();
    cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(tp2 - tp1).count() << "ms" << endl;
#ifdef DEBUG_MODE
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max memory used: " << usage.ru_maxrss << endl;
#endif
#ifdef DEBUG_WRITE
//    WriteDebugDataBispec(scmat, "debug_data/mat_bispec.txt");
    WriteDebugDataBispecAll(scmat, "debug_data/mat_bispec.txt");
#endif

#ifdef DEBUG_MODE
    //cout << scmat.size() << " : " << M3 << endl;
#endif

    return 0;
}
