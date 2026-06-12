#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include "l2l_transform.h"

using namespace std;

// Helper to convert integer Q10 bits into coeff_t
coeff_t int_to_coeff(int val) {
    double f_val = (double)val / 1024.0;
    return (coeff_t)f_val;
}

// Helper to convert coeff_t back to Q10 integer bits
int coeff_to_int(coeff_t val) {
    // Multiply by 1024 and round to nearest even
    double f_val = val.to_double() * 1024.0;
    return (int)round(f_val);
}

int main() {
    cout << "==================================================" << endl;
    cout << "Starting Vivado HLS C-Simulation for L2L Transform" << endl;
    cout << "==================================================" << endl;

    // Open golden vectors file
    ifstream infile("tb_vectors.txt");
    if (!infile.is_open()) {
        // Try alternate path for local dev testing
        infile.open("hardware/src/tb_vectors.txt");
        if (!infile.is_open()) {
            cerr << "Error: Could not open tb_vectors.txt" << endl;
            return 1;
        }
    }

    string tag;
    int num_blocks = 0;
    if (!(infile >> tag >> num_blocks) || tag != "NUM_BLOCKS") {
        cerr << "Error parsing NUM_BLOCKS" << endl;
        return 1;
    }
    cout << "Number of blocks to verify: " << num_blocks << endl;

    int errors = 0;

    for (int b = 0; b < num_blocks; b++) {
        int block_idx = -1;
        if (!(infile >> tag >> block_idx) || tag != "BLOCK" || block_idx != b) {
            cerr << "Error: Expected BLOCK " << b << ", got " << tag << " " << block_idx << endl;
            return 1;
        }
        cout << "\nVerifying Block " << b << "..." << endl;

        block_t x_n, s_prev, expected_y_n, expected_s_n;
        
        // Read X_N
        if (!(infile >> tag) || tag != "X_N:") {
            cerr << "Error parsing X_N tag" << endl;
            return 1;
        }
        for (int i = 0; i < M_BLOCK; i++) {
            for (int j = 0; j < M_BLOCK; j++) {
                int val;
                infile >> val;
                x_n[i][j] = int_to_coeff(val);
            }
        }

        // Read S_PREV
        if (!(infile >> tag) || tag != "S_PREV:") {
            cerr << "Error parsing S_PREV tag" << endl;
            return 1;
        }
        for (int i = 0; i < M_BLOCK; i++) {
            for (int j = 0; j < M_BLOCK; j++) {
                int val;
                infile >> val;
                s_prev[i][j] = int_to_coeff(val);
            }
        }

        // Read Y_N
        if (!(infile >> tag) || tag != "Y_N:") {
            cerr << "Error parsing Y_N tag" << endl;
            return 1;
        }
        for (int i = 0; i < M_BLOCK; i++) {
            for (int j = 0; j < M_BLOCK; j++) {
                int val;
                infile >> val;
                expected_y_n[i][j] = int_to_coeff(val);
            }
        }

        // Read S_N
        if (!(infile >> tag) || tag != "S_N:") {
            cerr << "Error parsing S_N tag" << endl;
            return 1;
        }
        for (int i = 0; i < M_BLOCK; i++) {
            for (int j = 0; j < M_BLOCK; j++) {
                int val;
                infile >> val;
                expected_s_n[i][j] = int_to_coeff(val);
            }
        }

        // Execute Forward Step
        block_t y_n_out, s_n_out;
        fwd_ladder_step_hls(x_n, s_prev, y_n_out, s_n_out);

        // Verify Forward outputs
        for (int i = 0; i < M_BLOCK; i++) {
            for (int j = 0; j < M_BLOCK; j++) {
                int y_out_val = coeff_to_int(y_n_out[i][j]);
                int y_exp_val = coeff_to_int(expected_y_n[i][j]);
                if (y_out_val != y_exp_val) {
                    cerr << "  Mismatch in Y_N [" << i << "][" << j << "]: got " << y_out_val << ", expected " << y_exp_val << endl;
                    errors++;
                }

                int s_out_val = coeff_to_int(s_n_out[i][j]);
                int s_exp_val = coeff_to_int(expected_s_n[i][j]);
                if (s_out_val != s_exp_val) {
                    cerr << "  Mismatch in S_N [" << i << "][" << j << "]: got " << s_out_val << ", expected " << s_exp_val << endl;
                    errors++;
                }
            }
        }

        // Execute Inverse Step (verify perfect reconstruction: s_n_out + y_n_out -> s_prev, x_n)
        block_t x_n_recon, s_prev_recon;
        inv_ladder_step_hls(y_n_out, s_n_out, x_n_recon, s_prev_recon);

        // Verify Perfect Reconstruction
        for (int i = 0; i < M_BLOCK; i++) {
            for (int j = 0; j < M_BLOCK; j++) {
                int x_rec_val = coeff_to_int(x_n_recon[i][j]);
                int x_orig_val = coeff_to_int(x_n[i][j]);
                if (x_rec_val != x_orig_val) {
                    cerr << "  Reconstruction Mismatch in X_N [" << i << "][" << j << "]: got " << x_rec_val << ", expected " << x_orig_val << endl;
                    errors++;
                }

                int s_prev_rec_val = coeff_to_int(s_prev_recon[i][j]);
                int s_prev_orig_val = coeff_to_int(s_prev[i][j]);
                if (s_prev_rec_val != s_prev_orig_val) {
                    cerr << "  Reconstruction Mismatch in S_PREV [" << i << "][" << j << "]: got " << s_prev_rec_val << ", expected " << s_prev_orig_val << endl;
                    errors++;
                }
            }
        }
    }

    infile.close();

    if (errors == 0) {
        cout << "\n==============================================" << endl;
        cout << "SUCCESS: All blocks verified bit-accurately!" << endl;
        cout << "Perfect reconstruction holds in HLS simulation." << endl;
        cout << "==============================================" << endl;
        return 0;
    } else {
        cerr << "\n==============================================" << endl;
        cerr << "FAILURE: Found " << errors << " bit-level mismatches." << endl;
        cerr << "==============================================" << endl;
        return 1;
    }
}
