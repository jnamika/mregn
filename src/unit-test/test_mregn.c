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
#include <string.h>
#include <math.h>
#include "mt19937ar.h"

#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "mregn.h"


/* assert functions */

void assert_consistency_mregn_state (
        struct mre_state *mre_s,
        struct rnn_state *gn_s,
        int gn_delay_length)
{
    int is_eq = 1;
    for (int n = gn_delay_length; n < gn_s->length && n < mre_s->length; n++) {
        for (int i = 0; i < mre_s->mre->expert_num; i++) {
            if (memcmp(gn_s->out_state[n-gn_delay_length]+i, mre_s->gate[i]+n,
                        sizeof(double))) {
                is_eq = 0;
            }
        }
    }
    mu_assert_with_msg(is_eq, "gn_s->out_state != mre_s->gate\n");
}

/* test functions */

static void test_mregn_forward_dynamics (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int gn_delay_length)
{
    for (int i = 0; i < gn->series_num; i++) {
        mregn_forward_dynamics(mre->mre_s + i, gn->rnn_s + i, gn_delay_length);
        assert_consistency_mregn_state(mre->mre_s + i, gn->rnn_s + i,
                gn_delay_length);
    }
}

static void test_mregn_forward_dynamics_forall (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int gn_delay_length)
{
    struct mixture_of_rnn_experts mre2;
    struct recurrent_neural_network gn2;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(mre, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_mixture_of_rnn_experts(&mre2, fp);

    fseek(fp, 0L, SEEK_SET);
    fwrite_recurrent_neural_network(gn, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_recurrent_neural_network(&gn2, fp);
    fclose(fp);

    mregn_forward_dynamics_forall(mre, gn, gn_delay_length);
    for (int i = 0; i < gn->series_num; i++) {
        mregn_forward_dynamics(mre2.mre_s + i, gn2.rnn_s + i, gn_delay_length);
        assert_equal_vector_sequence(mre2.mre_s[i].out_state,
                mre2.out_state_size * sizeof(double),
                mre2.mre_s[i].length, mre->mre_s[i].out_state,
                mre->out_state_size * sizeof(double), mre->mre_s[i].length);
        assert_consistency_mregn_state(mre->mre_s + i, gn->rnn_s + i,
                gn_delay_length);
    }

    free_recurrent_neural_network(&gn2);
    free_mixture_of_rnn_experts(&mre2);
}


static void test_mregn_forward_dynamics_in_closed_loop (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int mre_delay_length,
        int gn_delay_length)
{
    if (mre->in_state_size != mre->out_state_size && mre->in_state_size != 0) {
        return;
    }

    for (int i = 0; i < gn->series_num; i++) {
        mregn_forward_dynamics_in_closed_loop(mre->mre_s + i, gn->rnn_s + i,
                mre_delay_length, gn_delay_length);
        assert_consistency_mregn_state(mre->mre_s + i, gn->rnn_s + i,
                gn_delay_length);
    }
}


static void test_mregn_forward_dynamics_in_closed_loop_forall (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int mre_delay_length,
        int gn_delay_length)
{
    struct mixture_of_rnn_experts mre2;
    struct recurrent_neural_network gn2;
    FILE *fp;

    if (mre->in_state_size != mre->out_state_size && mre->in_state_size != 0) {
        return;
    }

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(mre, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_mixture_of_rnn_experts(&mre2, fp);

    fseek(fp, 0L, SEEK_SET);
    fwrite_recurrent_neural_network(gn, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_recurrent_neural_network(&gn2, fp);
    fclose(fp);

    mregn_forward_dynamics_in_closed_loop_forall(mre, gn, mre_delay_length,
            gn_delay_length);
    for (int i = 0; i < gn->series_num; i++) {
        mregn_forward_dynamics_in_closed_loop(mre2.mre_s + i, gn2.rnn_s + i,
                mre_delay_length, gn_delay_length);
        assert_equal_vector_sequence(mre2.mre_s[i].out_state,
                mre2.out_state_size * sizeof(double),
                mre2.mre_s[i].length, mre->mre_s[i].out_state,
                mre->out_state_size * sizeof(double), mre->mre_s[i].length);
        assert_consistency_mregn_state(mre->mre_s + i, gn->rnn_s + i,
                gn_delay_length);
    }

    free_mixture_of_rnn_experts(&mre2);
    free_recurrent_neural_network(&gn2);
}




void test_mre_state_setup (
        struct mixture_of_rnn_experts *mre,
        int target_num,
        int *target_length);

void test_gn_state_setup (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int gn_delay_length)
{
    const int expert_num = mre->expert_num;
    const int in_state_size = mre->in_state_size;
    for (int i = 0; i < mre->series_num; i++) {
        double **input, **target;
        MALLOC(input, mre->mre_s[i].length);
        MALLOC(input[0], mre->mre_s[i].length * (expert_num + in_state_size));
        for (int n = 0; n < mre->mre_s[i].length; n++) {
            input[n] = input[0] + (expert_num + in_state_size) * n;
            for (int j = 0; j < expert_num; j++) {
                input[n][j] = mre->mre_s[i].gate[j][n];
            }
            for (int j = 0; j < in_state_size; j++) {
                input[n][j+expert_num] =
                    mre->expert_rnn[0].rnn_s[i].in_state[n][j];
            }
        }
        target = input + gn_delay_length;
        rnn_add_target(gn, mre->mre_s[i].length - gn_delay_length, input,
                target);
        mre->mre_s[i].length -= gn_delay_length;
        for (int j = 0; j < expert_num; j++) {
            mre->expert_rnn[j].rnn_s[i].length -= gn_delay_length;
        }
        free(input[0]);
        free(input);
    }
}


static void test_mregn_setup (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        unsigned long seed,
        int expert_num,
        int in_state_size,
        int c_state_size,
        int out_state_size,
        int gn_c_state_size,
        int gn_delay_length,
        int target_num,
        int *target_length)
{
    init_genrand(seed);
    init_mixture_of_rnn_experts(mre, expert_num, in_state_size, c_state_size,
            out_state_size);
    init_recurrent_neural_network(gn, expert_num + in_state_size,
            gn_c_state_size, expert_num);

    test_mre_state_setup(mre, target_num, target_length);
    test_gn_state_setup(mre, gn, gn_delay_length);
}

void test_mregn (void)
{
    struct mixture_of_rnn_experts mre[4];
    struct recurrent_neural_network gn[4];
    int gn_delay_length[4] = {1, 2, 3, 1};

    test_mregn_setup(mre, gn, 4653L, 5, 3, 8, 4, 10, gn_delay_length[0], 2,
            (int[]){40,60});
    test_mregn_setup(mre+1, gn+1, 10951L, 8, 2, 10, 2, 10, gn_delay_length[1],
            1, (int[]){80});
    test_mregn_setup(mre+2, gn+2, 200L, 4, 4, 10, 2, 8, gn_delay_length[2], 3,
            (int[]){30,30,40});
    test_mregn_setup(mre+3, gn+3, 200L, 3, 0, 8, 2, 10, gn_delay_length[3], 3,
            (int[]){40,30,20});

    for (int i = 0; i < 4; i++) {
        mu_run_test_with_args(test_mregn_forward_dynamics, mre + i, gn + i,
                gn_delay_length[i]);
        mu_run_test_with_args(test_mregn_forward_dynamics_forall, mre + i, gn +
                i, gn_delay_length[i]);
        mu_run_test_with_args(test_mregn_forward_dynamics_in_closed_loop,
                mre + i, gn + i, gn_delay_length[i], 1);
        mu_run_test_with_args(test_mregn_forward_dynamics_in_closed_loop_forall,
                mre + i, gn + i, gn_delay_length[i], 1);
    }

    for (int i = 0; i < 4; i++) {
        free_mixture_of_rnn_experts(mre + i);
        free_recurrent_neural_network(gn + i);
    }
}


