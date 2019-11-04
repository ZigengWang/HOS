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
    in_data.erase(in_data.begin()); // erase if first value is id
}

void Reformat(vector<double>& in_data) {
    double sm = accumulate(in_data.begin(), in_data.end(), .0);
    double mean = sm / in_data.size();
    for (auto& d : in_data) d -= mean;
}

void ComputeFft(vector<double>& in_data, vector<complex<double>>& fdata) {
    int N = fdata.size();
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
    int N = fdata.size(), n = N - M3 + 1, MM3 = M3 + 1;
    double M3cu = 1.0 / (M3 * M3 * M3), div = 1.0 / N;
    vector<complex<double>> out(n * n);
    vector<vector<vector<complex<double>>>> scmat(MM3,
                                                  vector<vector<complex<double>>>(N,
                                                  vector<complex<double>>(N, (0, 0))));
#ifdef DEBUG_WRITE
    ofstream out_file("debug_data/mat_trispec_mem.txt");
#endif
    for (int k1 = 0; k1 < N; ++k1) {
        int k = k1 % MM3, ind = 0;
        for (int k2 = 0; k2 < N; ++k2) {
            for (int k3 = 0; k3 < N; ++k3) {
                scmat[k][k2][k3] = k2 <= k1 && k3 <= k2 && 2 * (k1 + k2 + k3) < N?
                                    fdata[k1] * fdata[k2] * fdata[k3] * conj(fdata[k1 + k2 + k3]) * div : .0;
                scmat[k][k2][k3] = scmat[k][k2][k3]
                                   + scmat[(k - 1 + MM3) % MM3][k2][k3]
                                   + (k2? scmat[k][k2 - 1][k3] : 0)
                                   + (k3? scmat[k][k2][k3 - 1] : 0)
                                   - (k2? scmat[(k - 1 + MM3) % MM3][k2 - 1][k3] : 0)
                                   - (k3? scmat[(k - 1 + MM3) % MM3][k2][k3 - 1] : 0)
                                   - (k2 && k3? scmat[k][k2 - 1][k3 - 1] : 0)
                                   + (k2 && k3? scmat[(k - 1 + MM3) % MM3][k2 - 1][k3 - 1] : 0);
                if (k1 + 1 >= M3 && k2 + 1 >= M3 && k3 + 1 >= M3) {
                    out[ind++] = M3cu * (scmat[k][k2][k3]
                                         - scmat[(k + 1) % MM3][k2][k3]
                                         - (k2 >= M3? scmat[k][k2 - M3][k3] : 0)
                                         - (k3 >= M3? scmat[k][k2][k3 - M3] : 0)
                                         + (k2 >= M3? scmat[(k + 1) % MM3][k2 - M3][k3] : 0)
                                         + (k3 >= M3? scmat[(k + 1) % MM3][k2][k3 - M3] : 0)
                                         + (k2 >= M3 && k3 >= M3? scmat[k][k2 - M3][k3 - M3] : 0)
                                         - (k2 >= M3 && k3 >= M3? scmat[(k + 1) % MM3][k2 - M3][k3 - M3] : 0));
                }
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
        cout << "\nusage: trispec_mem in_file_name\n" << endl;
        return EXIT_FAILURE;
    }

    string in_file_name(argv[1]);
    vector<double> in_data;
    ReadData(in_file_name, in_data); // read input time series data

    TimePoint tp1 = Clock::now();
    Reformat(in_data); // subtract mean from each value

    int N = in_data.size();
    vector<complex<double>> fdata(N);
    ComputeFft(in_data, fdata); // compute fft of the series
    int M3 = (int)pow(1.0 * N, .625);
    ComputeCumulantSmooth(fdata, M3);

    TimePoint tp2 = Clock::now();
    cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(tp2 - tp1).count() << "ms" << "\n";
#ifdef DEBUG_MODE
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max memory: " << usage.ru_maxrss << "KB" << "\n";
#endif
    return EXIT_SUCCESS;
}