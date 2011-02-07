/*
    Copyright (c) 2009-2011, Jun Namikawa <jnamika@gmail.com>

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "mregn.h"
#include "mregn_lyapunov.h"


/* assert functions */


/* test functions */

static void test_init_mregn_lyapunov_info (
        const struct mre_state *mre_s,
        const struct rnn_state *gn_s)
{
    struct mregn_lyapunov_info ml_info;
    assert_exit_call(init_mregn_lyapunov_info, &ml_info, mre_s, gn_s, 0, 1, 0);
    assert_exit_call(init_mregn_lyapunov_info, &ml_info, mre_s, gn_s, 1, 0, 0);
    assert_exit_call(init_mregn_lyapunov_info, &ml_info, mre_s, gn_s, 1, 1, -1);
    assert_exit_call(init_mregn_lyapunov_info, &ml_info, mre_s, gn_s, 1, 1,
            gn_s->length);
    if (mre_s->mre->in_state_size == mre_s->mre->out_state_size ||
            mre_s->mre->in_state_size == 0) {
        assert_exit_nocall(init_mregn_lyapunov_info, &ml_info, mre_s, gn_s,
                1, 1, 0);
        free_mregn_lyapunov_info(&ml_info);
    }
}



double** mre_jacobian_for_lyapunov_spectrum (const double* vector, int n, int t,
        double** matrix, void *obj);

static void test_mre_jacobian_for_lyapunov_spectrum (
        struct mre_state *mre_s,
        struct rnn_state *gn_s)
{
    if (mre_s->mre->in_state_size != mre_s->mre->out_state_size &&
            mre_s->mre->in_state_size != 0) {
        return;
    }

    double **matrix;
    struct mregn_lyapunov_info ml_info;
    init_mregn_lyapunov_info(&ml_info, mre_s, gn_s, 1, 1, 0);
    MALLOC2(matrix, ml_info.dimension, ml_info.dimension);

    assert_exit_call(mre_jacobian_for_lyapunov_spectrum, NULL,
            ml_info.dimension, 0, matrix, &ml_info);
    assert_exit_call(mre_jacobian_for_lyapunov_spectrum, ml_info.state[1],
            ml_info.dimension, 0, matrix, &ml_info);
    assert_exit_call(mre_jacobian_for_lyapunov_spectrum, ml_info.state[0],
            ml_info.dimension+1, 0, matrix, &ml_info);
    assert_exit_call(mre_jacobian_for_lyapunov_spectrum, ml_info.state[0],
            ml_info.dimension-1, 0, matrix, &ml_info);
    assert_exit_call(mre_jacobian_for_lyapunov_spectrum, ml_info.state[0],
            ml_info.dimension, -1, matrix, &ml_info);
    assert_exit_call(mre_jacobian_for_lyapunov_spectrum, ml_info.state[0],
            ml_info.dimension, gn_s->length, matrix, &ml_info);
    assert_exit_call(mre_jacobian_for_lyapunov_spectrum, ml_info.state[0],
            ml_info.dimension, mre_s->length, matrix, &ml_info);

    int min_length = (gn_s->length < mre_s->length) ?
        gn_s->length : mre_s->length;
    mu_assert(mre_jacobian_for_lyapunov_spectrum(ml_info.state[0],
                ml_info.dimension, 0, matrix, &ml_info) != NULL);
    mu_assert(mre_jacobian_for_lyapunov_spectrum(ml_info.state[min_length-1],
                ml_info.dimension, min_length-1, matrix, &ml_info) != NULL);

    FREE2(matrix);
    free_mregn_lyapunov_info(&ml_info);
}


static void test_mregn_lyapunov_spectrum (
        struct mre_state *mre_s,
        struct rnn_state *gn_s)
{
    if (mre_s->mre->in_state_size != mre_s->mre->out_state_size &&
            mre_s->mre->in_state_size != 0) {
        return;
    }

    struct mregn_lyapunov_info ml_info;
    init_mregn_lyapunov_info(&ml_info, mre_s, gn_s, 1, 1, 0);

    double spectrum[ml_info.dimension];
    mregn_forward_dynamics_in_closed_loop(mre_s, gn_s, ml_info.mre_delay_length,
            ml_info.gn_delay_length);
    reset_mregn_lyapunov_info(&ml_info);
    mregn_lyapunov_spectrum(&ml_info, spectrum, ml_info.dimension);
    mu_assert(spectrum[0] < 0);
    for (int i = 1; i < ml_info.dimension; i++) {
        mu_assert(spectrum[i-1] >= spectrum[i]);
    }
    free_mregn_lyapunov_info(&ml_info);
}





void test_mre_state_setup (
        struct mixture_of_rnn_experts *mre,
        int target_num,
        int *target_length);

void test_gn_state_setup (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int gn_delay_length);

static void test_mregn_lyapunov_setup(
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        unsigned long seed,
        int expert_num,
        int in_state_size,
        int c_state_size,
        int out_state_size,
        int gn_c_state_size,
        int gn_delay_length)
{
    init_genrand(seed);
    init_mixture_of_rnn_experts(mre, expert_num, in_state_size, c_state_size,
            out_state_size);
    init_recurrent_neural_network(gn, expert_num + in_state_size,
            gn_c_state_size, expert_num);

    test_mre_state_setup(mre, 1, (int[]){100});
    test_gn_state_setup(mre, gn, gn_delay_length);
}


void test_mregn_lyapunov (void)
{
    struct mixture_of_rnn_experts mre[4];
    struct recurrent_neural_network gn[4];
    test_mregn_lyapunov_setup(mre, gn, 4653L, 3, 4, 6, 4, 8, 1);
    test_mregn_lyapunov_setup(mre+1, gn+1, 99043L, 5, 0, 5, 3, 4, 2);
    test_mregn_lyapunov_setup(mre+2, gn+2, 99043L, 5, 2, 5, 2, 3, 3);
    test_mregn_lyapunov_setup(mre+3, gn+3, 99043L, 5, 8, 5, 3, 5, 2);

    for (int i = 0; i < 4; i++) {
        mu_run_test_with_args(test_init_mregn_lyapunov_info, mre[i].mre_s,
                gn[i].rnn_s);
        mu_run_test_with_args(test_mre_jacobian_for_lyapunov_spectrum,
                mre[i].mre_s, gn[i].rnn_s);
        mu_run_test_with_args(test_mregn_lyapunov_spectrum, mre[i].mre_s,
                gn[i].rnn_s);
        free_mixture_of_rnn_experts(mre+i);
        free_recurrent_neural_network(gn+i);
    }

}


