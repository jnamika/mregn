/*
    Copyright (c) 2010-2011, Jun Namikawa <jnamika@gmail.com>

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
#include "mregn_runner.h"


/* assert functions */

void assert_equal_rnn_p (
        struct rnn_parameters *rnn1_p,
        struct rnn_parameters *rnn2_p);

void assert_equal_rnn_s (
        struct rnn_state *rnn1_s,
        struct rnn_state *rnn2_s);

void assert_equal_mre_s (
        struct mre_state *mre1_s,
        struct mre_state *mre2_s);




/* test functions */

static void test_new_mregn_runner (void)
{
    struct mregn_runner *runner;
    int stat = _new_mregn_runner(&runner);
    assert_equal_int(0, stat);
    mu_assert(runner != NULL);
    _delete_mregn_runner(runner);
}



typedef struct test_mregn_runner_data {
    struct mregn_runner runner;
    int expert_num;
    int in_state_size;
    int c_state_size;
    int out_state_size;
    int gn_c_state_size;
    int mre_delay_length;
    int gn_delay_length;
    int target_num;
} test_mregn_runner_data;



void test_mre_state_setup (
        struct mixture_of_rnn_experts *mre,
        int target_num,
        int *target_length);

void test_gn_state_setup (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int gn_delay_length);

static void test_init_mregn_runner (
        struct test_mregn_runner_data *t_data,
        int expert_num,
        int in_state_size,
        int c_state_size,
        int out_state_size,
        int gn_c_state_size,
        int mre_delay_length,
        int gn_delay_length,
        int target_num,
        int *target_length)
{
    struct mixture_of_rnn_experts mre;
    struct recurrent_neural_network gn;
    FILE *mre_fp, *gn_fp;

    init_mixture_of_rnn_experts(&mre, expert_num, in_state_size, c_state_size,
            out_state_size);
    init_recurrent_neural_network(&gn, expert_num + in_state_size,
            gn_c_state_size, expert_num);

    test_mre_state_setup(&mre, target_num, target_length);
    test_gn_state_setup(&mre, &gn, gn_delay_length);

    mre_fp = tmpfile();
    gn_fp = tmpfile();
    if (mre_fp == NULL || gn_fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    if (fwrite(&mre_delay_length, sizeof(int), 1, mre_fp) != 1) {
        print_error_msg();
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(&mre, mre_fp);
    if (fwrite(&gn_delay_length, sizeof(int), 1, gn_fp) != 1) {
        print_error_msg();
        exit(EXIT_FAILURE);
    }
    fwrite_recurrent_neural_network(&gn, gn_fp);
    fseek(mre_fp, 0L, SEEK_SET);
    fseek(gn_fp, 0L, SEEK_SET);
    init_mregn_runner(&t_data->runner, mre_fp, gn_fp);
    fclose(mre_fp);
    fclose(gn_fp);

    t_data->expert_num = expert_num;
    t_data->in_state_size = in_state_size;
    t_data->c_state_size = c_state_size;
    t_data->out_state_size = out_state_size;
    t_data->gn_c_state_size = gn_c_state_size;
    t_data->mre_delay_length = mre_delay_length;
    t_data->gn_delay_length = gn_delay_length;
    t_data->target_num = target_num;

    assert_equal_int(expert_num,
            gn_out_state_size_from_runner(&t_data->runner));
    assert_equal_int(in_state_size,
            mre_in_state_size_from_runner(&t_data->runner));
    for (int i = 0; i < mre.expert_num; i++) {
        assert_equal_int(c_state_size,
                expert_rnn_c_state_size_from_runner(&t_data->runner, i));
    }
    assert_equal_int(out_state_size,
            mre_out_state_size_from_runner(&t_data->runner));
    assert_equal_int(gn_c_state_size,
            gn_c_state_size_from_runner(&t_data->runner));
    assert_equal_int(mre_delay_length,
            mre_delay_length_from_runner(&t_data->runner));
    assert_equal_int(gn_delay_length,
            gn_delay_length_from_runner(&t_data->runner));
    assert_equal_int(target_num, gn_target_num_from_runner(&t_data->runner));
    assert_equal_rnn_p(&gn.rnn_p, &t_data->runner.gn.rnn_p);
    for (int i = 0; i < target_num; i++) {
        assert_equal_rnn_s(gn.rnn_s + i, t_data->runner.gn.rnn_s + i);
        assert_equal_mre_s(mre.mre_s + i, t_data->runner.mre.mre_s + i);
    }
    free_recurrent_neural_network(&gn);
    free_mixture_of_rnn_experts(&mre);
}


static void test_set_init_state_of_mregn_runner (
        struct test_mregn_runner_data *t_data)
{
    struct mregn_runner *runner = &t_data->runner;
    assert_exit_nocall(set_init_state_of_mregn_runner, runner, -1);
    assert_exit_nocall(set_init_state_of_mregn_runner, runner, 0);
    assert_exit_nocall(set_init_state_of_mregn_runner, runner, 1000);

    for (int i = 0; i < t_data->target_num; i++) {
        set_init_state_of_mregn_runner(runner, i);
        int length = gn_delay_length_from_runner(runner);
        if (length > runner->gn.rnn_s[i].length) {
            length = runner->gn.rnn_s[i].length;
        }
        const struct rnn_state *g_dst = gn_state_from_runner(runner);
        const struct rnn_state *g_src = runner->gn.rnn_s + i;
        assert_equal_vector_sequence(g_src->in_state,
                g_src->rnn_p->in_state_size, length, g_dst->in_state,
                g_dst->rnn_p->in_state_size, length);
        assert_equal_memory(g_src->init_c_state, g_src->rnn_p->c_state_size *
                sizeof(double), g_dst->init_c_state, g_dst->rnn_p->c_state_size
                * sizeof(double));
        assert_equal_memory(g_src->init_c_inter_state,
                g_src->rnn_p->c_state_size * sizeof(double),
                g_dst->init_c_inter_state, g_dst->rnn_p->c_state_size *
                sizeof(double));

        length = mre_delay_length_from_runner(runner);
        if (length > runner->mre.mre_s[i].length) {
            length = runner->mre.mre_s[i].length;
        }
        const struct mre_state *m_dst = mre_state_from_runner(runner);
        const struct mre_state *m_src = runner->mre.mre_s + i;
        for (int j = 0; j < runner->mre.expert_num; j++) {
            const struct rnn_state *e_dst = m_dst->expert_rnn_s[j];
            const struct rnn_state *e_src = m_src->expert_rnn_s[j];
            assert_equal_vector_sequence(e_src->in_state,
                    e_src->rnn_p->in_state_size, length, e_dst->in_state,
                    e_dst->rnn_p->in_state_size, length);
            assert_equal_memory(e_src->init_c_state, e_src->rnn_p->c_state_size
                    * sizeof(double), e_dst->init_c_state,
                    e_dst->rnn_p->c_state_size * sizeof(double));
            assert_equal_memory(e_src->init_c_inter_state,
                    e_src->rnn_p->c_state_size * sizeof(double),
                    e_dst->init_c_inter_state, e_dst->rnn_p->c_state_size *
                    sizeof(double));
        }
    }
}

static void test_update_mregn_runner (struct test_mregn_runner_data *t_data)
{
    struct mregn_runner *runner = &t_data->runner;
    const int expert_mem_size = t_data->expert_num * sizeof(double);
    const int c_mem_size = t_data->c_state_size * sizeof(double);
    const int out_mem_size = t_data->out_state_size * sizeof(double);
    const int gn_c_mem_size = t_data->gn_c_state_size * sizeof(double);
    for (int i = 0; i < t_data->target_num; i++) {
        set_init_state_of_mregn_runner(runner, i);
        struct rnn_state *gn_s = runner->gn.rnn_s + i;
        struct mre_state *mre_s = runner->mre.mre_s + i;
        mregn_forward_dynamics_in_closed_loop(mre_s, gn_s,
                t_data->mre_delay_length, t_data->gn_delay_length);
        for (int n = 0; n < gn_s->length; n++) {
            update_mregn_runner(runner);
            assert_equal_memory(gn_s->out_state[n], expert_mem_size,
                    gn_out_state_from_runner(runner), expert_mem_size);
            assert_equal_memory(gn_s->c_state[n], gn_c_mem_size,
                    gn_c_state_from_runner(runner), gn_c_mem_size);
            assert_equal_memory(gn_s->c_inter_state[n], gn_c_mem_size,
                    gn_c_inter_state_from_runner(runner), gn_c_mem_size);

            assert_equal_memory(mre_s->out_state[n], out_mem_size,
                    mre_out_state_from_runner(runner), out_mem_size);
            for (int j = 0; j < mre_s->mre->expert_num; j++) {
                assert_equal_memory(mre_s->expert_rnn_s[j]->c_state[n],
                        c_mem_size, expert_rnn_c_state_from_runner(runner, j),
                        c_mem_size);
                assert_equal_memory(mre_s->expert_rnn_s[j]->c_inter_state[n],
                        c_mem_size,
                        expert_rnn_c_inter_state_from_runner(runner, j),
                        c_mem_size);
            }
        }
    }
}

static void test_gn_in_state_size_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->expert_num + t_data->in_state_size,
            gn_in_state_size_from_runner(&t_data->runner));
}

static void test_gn_c_state_size_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->gn_c_state_size,
            gn_c_state_size_from_runner(&t_data->runner));
}

static void test_gn_out_state_size_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->expert_num,
            gn_out_state_size_from_runner(&t_data->runner));
}

static void test_gn_delay_length_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->gn_delay_length,
            gn_delay_length_from_runner(&t_data->runner));
}

static void test_gn_target_num_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->target_num,
            gn_target_num_from_runner(&t_data->runner));
}

static void test_gn_in_state_from_runner (struct test_mregn_runner_data *t_data)
{
    const struct rnn_state *gn_s = gn_state_from_runner(&t_data->runner);
    assert_equal_pointer(gn_s->in_state[0],
            gn_in_state_from_runner(&t_data->runner));
}

static void test_gn_c_state_from_runner (struct test_mregn_runner_data *t_data)
{
    const struct rnn_state *gn_s = gn_state_from_runner(&t_data->runner);
    assert_equal_pointer(gn_s->init_c_state,
            gn_c_state_from_runner(&t_data->runner));
}

static void test_gn_c_inter_state_from_runner (
        struct test_mregn_runner_data *t_data)
{
    const struct rnn_state *gn_s = gn_state_from_runner(&t_data->runner);
    assert_equal_pointer(gn_s->init_c_inter_state,
            gn_c_inter_state_from_runner(&t_data->runner));
}

static void test_gn_out_state_from_runner (
        struct test_mregn_runner_data *t_data)
{
    const struct rnn_state *gn_s = gn_state_from_runner(&t_data->runner);
    assert_equal_pointer(gn_s->out_state[0],
            gn_out_state_from_runner(&t_data->runner));
}

static void test_gn_state_from_runner (struct test_mregn_runner_data *t_data)
{
    const struct rnn_state *gn_s = t_data->runner.gn.rnn_s + t_data->runner.id;
    assert_equal_pointer(gn_s, gn_state_from_runner(&t_data->runner));
}

static void test_mre_in_state_size_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->in_state_size,
            mre_in_state_size_from_runner(&t_data->runner));
}

static void test_mre_out_state_size_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->out_state_size,
            mre_out_state_size_from_runner(&t_data->runner));
}

static void test_mre_delay_length_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->mre_delay_length,
            mre_delay_length_from_runner(&t_data->runner));
}

static void test_mre_target_num_from_runner (
        struct test_mregn_runner_data *t_data)
{
    assert_equal_int(t_data->target_num,
            mre_target_num_from_runner(&t_data->runner));
}

static void test_mre_in_state_from_runner (
        struct test_mregn_runner_data *t_data)
{
    const struct mre_state *mre_s = mre_state_from_runner(&t_data->runner);
    assert_equal_pointer(mre_s->expert_rnn_s[0]->in_state[0],
            mre_in_state_from_runner(&t_data->runner));
}

static void test_mre_out_state_from_runner (
        struct test_mregn_runner_data *t_data)
{
    const struct mre_state *mre_s = mre_state_from_runner(&t_data->runner);
    assert_equal_pointer(mre_s->out_state[0],
            mre_out_state_from_runner(&t_data->runner));
}

static void test_mre_state_from_runner (struct test_mregn_runner_data *t_data)
{
    const struct mre_state *mre_s = t_data->runner.mre.mre_s +
        t_data->runner.id;
    assert_equal_pointer(mre_s, mre_state_from_runner(&t_data->runner));
}

static void test_expert_rnn_c_state_size_from_runner (
        struct test_mregn_runner_data *t_data)
{
    const int expert_num = t_data->expert_num;
    for (int i = 0; i < expert_num; i++) {
        assert_equal_int(t_data->c_state_size,
                expert_rnn_c_state_size_from_runner(&t_data->runner, i));
    }
}

static void test_expert_rnn_c_state_from_runner (
        struct test_mregn_runner_data *t_data)
{
    const int expert_num = t_data->expert_num;
    for (int i = 0; i < expert_num; i++) {
        const struct rnn_state *rnn_s =
            expert_rnn_state_from_runner(&t_data->runner, i);
        assert_equal_pointer(rnn_s->init_c_state,
                expert_rnn_c_state_from_runner(&t_data->runner, i));
    }
}

static void test_expert_rnn_c_inter_state_from_runner (
        struct test_mregn_runner_data *t_data)
{
    const int expert_num = t_data->expert_num;
    for (int i = 0; i < expert_num; i++) {
        const struct rnn_state *rnn_s =
            expert_rnn_state_from_runner(&t_data->runner, i);
        assert_equal_pointer(rnn_s->init_c_inter_state,
                expert_rnn_c_inter_state_from_runner(&t_data->runner, i));
    }
}

static void test_expert_rnn_state_from_runner (
        struct test_mregn_runner_data *t_data)
{
    const int expert_num = t_data->expert_num;
    const struct mre_state *mre_s = t_data->runner.mre.mre_s +
        t_data->runner.id;
    for (int i = 0; i < expert_num; i++) {
        const struct rnn_state *rnn_s = mre_s->expert_rnn_s[i];
        assert_equal_pointer(rnn_s,
                expert_rnn_state_from_runner(&t_data->runner, i));
    }
}



void test_mregn_runner (void)
{
    struct test_mregn_runner_data t_data[4];

    init_genrand(801759L);

    mu_run_test(test_new_mregn_runner);

    mu_run_test_with_args(test_init_mregn_runner, t_data, 3, 1, 10, 1, 10, 1, 1,
            2, (int[]){50,100});
    mu_run_test_with_args(test_init_mregn_runner, t_data+1, 1, 3, 13, 3, 15, 2,
            3, 3, (int[]){30,30,20});
    mu_run_test_with_args(test_init_mregn_runner, t_data+2, 2, 0, 7, 2, 8, 3, 5,
            2, (int[]){100,50});
    mu_run_test_with_args(test_init_mregn_runner, t_data+3, 8, 4, 10, 4, 10, 20,
            12, 3, (int[]){50,50,50});

    for (int i = 0; i < 4; i++) {
        mu_run_test_with_args(test_set_init_state_of_mregn_runner, t_data + i);
        mu_run_test_with_args(test_update_mregn_runner, t_data + i);
        mu_run_test_with_args(test_gn_in_state_size_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_c_state_size_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_out_state_size_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_delay_length_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_target_num_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_in_state_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_c_state_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_c_inter_state_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_out_state_from_runner, t_data + i);
        mu_run_test_with_args(test_gn_state_from_runner, t_data + i);
        mu_run_test_with_args(test_mre_in_state_size_from_runner, t_data + i);
        mu_run_test_with_args(test_mre_out_state_size_from_runner, t_data + i);
        mu_run_test_with_args(test_mre_delay_length_from_runner, t_data + i);
        mu_run_test_with_args(test_mre_target_num_from_runner, t_data + i);
        mu_run_test_with_args(test_mre_in_state_from_runner, t_data + i);
        mu_run_test_with_args(test_mre_out_state_from_runner, t_data + i);
        mu_run_test_with_args(test_mre_state_from_runner, t_data + i);
        mu_run_test_with_args(test_expert_rnn_c_state_size_from_runner,
                t_data + i);
        mu_run_test_with_args(test_expert_rnn_c_state_from_runner, t_data + i);
        mu_run_test_with_args(test_expert_rnn_c_inter_state_from_runner,
                t_data + i);
        mu_run_test_with_args(test_expert_rnn_state_from_runner, t_data + i);

        free_mregn_runner(&t_data[i].runner);
    }
}


