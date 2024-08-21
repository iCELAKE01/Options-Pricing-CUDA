#include <iostream>
#include <cmath>

using namespace std;

// Black-Scholes formula
float blackScholes(float S, float K, float t, float r, float v) {
    float d1 = (log(S / K) + (r + 0.5 * v * v) * t) / (v * sqrt(t));
    float d2 = d1 - v * sqrt(t);
    float N1 = 0.5 + (1.0 / sqrt(2.0 * M_PI)) * exp(-d1 * d1 / 2.0);
    float N2 = 0.5 + (1.0 / sqrt(2.0 * M_PI)) * exp(-d2 * d2 / 2.0);
    float price = S * N1 - K * exp(-r * t) * N2;
    return price;
}

int main() {
    // Initialize input data
    int numOptions = 1000000;
    float* prices = new float[numOptions];
    float* strikes = new float[numOptions];
    float* times = new float[numOptions];
    float* risks = new float[numOptions];
    float* vols = new float[numOptions];

    // Initialize output data
    float* results = new float[numOptions];

    // Calculate prices
    for (int i = 0; i < numOptions; i++) {
        results[i] = blackScholes(prices[i], strikes[i], times[i], risks[i], vols[i]);
    }

    // Print results
    for (int i = 0; i < numOptions; i++) {
        cout << "Option " << i << ": " << results[i] << endl;
    }

    // Clean up
    delete[] prices;
    delete[] strikes;
    delete[] times;
    delete[] risks;
    delete[] vols;
    delete[] results;

    return 0;
}