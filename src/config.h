#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <chrono>
using namespace std;

#define DEBUG_MODE
#define DEBUG_WRITE

using Clock = chrono::system_clock;
using TimePoint = chrono::system_clock::time_point;

const double threshold = 1e-6;
#endif
