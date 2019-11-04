/*
 *
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
    vector<complex<double>> out(n);
    vector<vector<complex<double>>> scmat(MM3, vector<complex<double>>(N, (0, 0)));
#ifdef DEBUG_WRITE
    ofstream out_file("debug_data/mat_bispec_mem.txt");
#endif
    for (int k1 = 0; k1 < N; ++k1) {
        int k = k1 % MM3, ind = 0;
        for (int k2 = 0; k2 < N; ++k2) {
            scmat[k][k2] = k2 <= k1 && 2 * (k1 + k2) < N? (fdata[k1] * fdata[k2] * conj(fdata[k1 + k2])) * div : .0;
            scmat[k][k2] = scmat[k][k2] + scmat[(k - 1 + MM3) % MM3][k2]
                           + (k2? scmat[k][k2 - 1] - scmat[(k - 1 + MM3) % MM3][k2 - 1] : 0);
            if (k1 + 1 >= M3 && k2 + 1 >= M3) {
                out[ind++] = M3sq * (scmat[k][k2] - scmat[(k + 1) % MM3][k2] +
                             (k2 >= M3? -scmat[k][k2 - M3] + scmat[(k + 1) % MM3][k2 - M3] : 0));
            }
        }
#ifdef DEBUG_WRITE
        if (ind) {
            //cout << k1 << " " << ind << endl;
            for (int i = 0; i < ind; ++i) {
                if (abs(out[i].real()) < threshold) out[i].real(0);
                if (abs(out[i].imag()) < threshold) out[i].imag(0);
                if (abs(out[i])) {
                    //out_file << k1 << " " << i << ": " << out[i] << "\n";
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
        cout << "\nusage: bispec_mem in_file_name\n" << endl;
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