#include "l2l_transform.h"

// 2D DCT-II via matrix multiplication: out = C * in * C^T
void dct_2d_hls(block_t in, block_t out) {
    #pragma HLS ARRAY_PARTITION variable=in complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out complete dim=1

    coeff_t C[M_BLOCK][M_BLOCK];
    coeff_t D[M_BLOCK][M_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=C complete dim=0
    #pragma HLS ARRAY_PARTITION variable=D complete dim=0

    // Initialize coefficients
    init_coeffs: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS UNROLL
            C[i][j] = (coeff_t)DCT_COEFFS[i][j];
            D[i][j] = (coeff_t)IDCT_COEFFS[i][j];
        }
    }

    coeff_t temp[M_BLOCK][M_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=temp complete dim=1

    // First matrix multiplication: temp = C * in
    // temp[i][j] = sum_{k=0..7} C[i][k] * in[k][j]
    dct_mult1_outer: for (int i = 0; i < M_BLOCK; i++) {
        dct_mult1_inner: for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 10, AP_TRN, AP_RND_CONV> sum = 0;
            dct_mult1_accum: for (int k = 0; k < M_BLOCK; k++) {
                #pragma HLS UNROLL
                sum += C[i][k] * in[k][j];
            }
            temp[i][j] = (coeff_t)sum;
        }
    }

    // Second matrix multiplication: out = temp * C^T = temp * D
    // out[i][j] = sum_{k=0..7} temp[i][k] * D[k][j]
    dct_mult2_outer: for (int i = 0; i < M_BLOCK; i++) {
        dct_mult2_inner: for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 10, AP_TRN, AP_RND_CONV> sum = 0;
            dct_mult2_accum: for (int k = 0; k < M_BLOCK; k++) {
                #pragma HLS UNROLL
                sum += temp[i][k] * D[k][j];
            }
            out[i][j] = (coeff_t)sum;
        }
    }
}

// 2D IDCT-III via matrix multiplication: out = D * in * D^T
void idct_2d_hls(block_t in, block_t out) {
    #pragma HLS ARRAY_PARTITION variable=in complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out complete dim=1

    coeff_t C[M_BLOCK][M_BLOCK];
    coeff_t D[M_BLOCK][M_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=C complete dim=0
    #pragma HLS ARRAY_PARTITION variable=D complete dim=0

    // Initialize coefficients
    init_coeffs: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS UNROLL
            C[i][j] = (coeff_t)DCT_COEFFS[i][j];
            D[i][j] = (coeff_t)IDCT_COEFFS[i][j];
        }
    }

    coeff_t temp[M_BLOCK][M_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=temp complete dim=1

    // First matrix multiplication: temp = D * in
    // temp[i][j] = sum_{k=0..7} D[i][k] * in[k][j]
    idct_mult1_outer: for (int i = 0; i < M_BLOCK; i++) {
        idct_mult1_inner: for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 10, AP_TRN, AP_RND_CONV> sum = 0;
            idct_mult1_accum: for (int k = 0; k < M_BLOCK; k++) {
                #pragma HLS UNROLL
                sum += D[i][k] * in[k][j];
            }
            temp[i][j] = (coeff_t)sum;
        }
    }

    // Second matrix multiplication: out = temp * D^T = temp * C
    // out[i][j] = sum_{k=0..7} temp[i][k] * C[k][j]
    idct_mult2_outer: for (int i = 0; i < M_BLOCK; i++) {
        idct_mult2_inner: for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 10, AP_TRN, AP_RND_CONV> sum = 0;
            idct_mult2_accum: for (int k = 0; k < M_BLOCK; k++) {
                #pragma HLS UNROLL
                sum += temp[i][k] * C[k][j];
            }
            out[i][j] = (coeff_t)sum;
        }
    }
}

// Top-level IP block for forward block-ladder step
void fwd_ladder_step_hls(block_t x_n, block_t s_prev, block_t y_n, block_t s_n) {
    #pragma HLS ARRAY_PARTITION variable=x_n complete dim=1
    #pragma HLS ARRAY_PARTITION variable=s_prev complete dim=1
    #pragma HLS ARRAY_PARTITION variable=y_n complete dim=1
    #pragma HLS ARRAY_PARTITION variable=s_n complete dim=1

    block_t idct_s;
    block_t u1;
    block_t dct_u1;
    block_t u2;
    block_t idct_yn;
    #pragma HLS ARRAY_PARTITION variable=idct_s complete dim=1
    #pragma HLS ARRAY_PARTITION variable=u1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dct_u1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=u2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=idct_yn complete dim=1

    // Step 1: idct_s = IDCT2D(s_prev)
    idct_2d_hls(s_prev, idct_s);

    // u1 = x_n - idct_s
    step1_loop: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            u1[i][j] = x_n[i][j] - idct_s[i][j];
        }
    }

    // Step 2: dct_u1 = DCT2D(u1)
    dct_2d_hls(u1, dct_u1);

    // u2 = s_prev + (dct_u1 - u1)
    step2_loop: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            u2[i][j] = s_prev[i][j] + (dct_u1[i][j] - u1[i][j]);
        }
    }

    // Step 3: y_n = u1 + u2
    step3_loop: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            y_n[i][j] = u1[i][j] + u2[i][j];
        }
    }

    // Step 4: idct_yn = IDCT2D(y_n)
    idct_2d_hls(y_n, idct_yn);

    // s_n = u2 + (idct_yn - y_n)
    step4_loop: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            s_n[i][j] = u2[i][j] + (idct_yn[i][j] - y_n[i][j]);
        }
    }
}

// Top-level IP block for inverse block-ladder step
void inv_ladder_step_hls(block_t y_n, block_t s_n, block_t x_n, block_t s_prev) {
    #pragma HLS ARRAY_PARTITION variable=y_n complete dim=1
    #pragma HLS ARRAY_PARTITION variable=s_n complete dim=1
    #pragma HLS ARRAY_PARTITION variable=x_n complete dim=1
    #pragma HLS ARRAY_PARTITION variable=s_prev complete dim=1

    block_t idct_yn;
    block_t u2;
    block_t u1;
    block_t dct_u1;
    block_t idct_s_prev;
    #pragma HLS ARRAY_PARTITION variable=idct_yn complete dim=1
    #pragma HLS ARRAY_PARTITION variable=u2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=u1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dct_u1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=idct_s_prev complete dim=1

    // Step 1: idct_yn = IDCT2D(y_n)
    idct_2d_hls(y_n, idct_yn);

    // u2 = s_n - (idct_yn - y_n)
    step1_loop: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            u2[i][j] = s_n[i][j] - (idct_yn[i][j] - y_n[i][j]);
        }
    }

    // Step 2: u1 = y_n - u2
    step2_loop: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            u1[i][j] = y_n[i][j] - u2[i][j];
        }
    }

    // Step 3: dct_u1 = DCT2D(u1)
    dct_2d_hls(u1, dct_u1);

    // s_prev = u2 - (dct_u1 - u1)
    step3_loop: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            s_prev[i][j] = u2[i][j] - (dct_u1[i][j] - u1[i][j]);
        }
    }

    // Step 4: idct_s_prev = IDCT2D(s_prev)
    idct_2d_hls(s_prev, idct_s_prev);

    // x_n = u1 + idct_s_prev
    step4_loop: for (int i = 0; i < M_BLOCK; i++) {
        for (int j = 0; j < M_BLOCK; j++) {
            #pragma HLS PIPELINE II=1
            x_n[i][j] = u1[i][j] + idct_s_prev[i][j];
        }
    }
}
