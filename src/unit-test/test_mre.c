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

#define TEST_CODE
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "mre.h"


#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif


/* assert functions */

void assert_equal_mre_s (
        struct mre_state *mre1_s,
        struct mre_state *mre2_s)
{
    assert_equal_int(mre1_s->length, mre2_s->length);

    assert_equal_vector_sequence(mre1_s->gate, mre1_s->length * sizeof(double),
            mre1_s->mre->expert_num, mre2_s->gate,
            mre2_s->length * sizeof(double), mre2_s->mre->expert_num);
    assert_equal_vector_sequence(mre1_s->beta, mre1_s->length * sizeof(double),
            mre1_s->mre->expert_num, mre2_s->beta,
            mre2_s->length * sizeof(double), mre2_s->mre->expert_num);
    assert_equal_vector_sequence(mre1_s->delta_beta,
            mre1_s->length * sizeof(double), mre1_s->mre->expert_num,
            mre2_s->delta_beta, mre2_s->length * sizeof(double),
            mre2_s->mre->expert_num);
}


void assert_equal_rnn_p (
        struct rnn_parameters *rnn1_p,
        struct rnn_parameters *rnn2_p);

void assert_equal_rnn_s (
        struct rnn_state *rnn1_s,
        struct rnn_state *rnn2_s);


#ifdef ENABLE_ATTRACTION_OF_INIT_C
static double get_posterior_distribution (
        struct mixture_of_rnn_experts *mre,
        double ***mean,
        double ***variance)
#else
static double get_posterior_distribution (struct mixture_of_rnn_experts *mre)
#endif
{
    double post_dist = 0;
    for (int i = 0; i < mre->series_num; i++) {
        mre_set_likelihood(mre->mre_s + i);
        post_dist += mre_get_joint_likelihood(mre->mre_s + i);
        post_dist += mre_get_prior_likelihood(mre->mre_s + i);
    }
    for (int i = 0; i < mre->expert_num; i++) {
        struct recurrent_neural_network *rnn = mre->expert_rnn + i;
        double d = rnn->rnn_p.prior_sigma - rnn->rnn_p.sigma;
        post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            for (int k = 0; k < rnn->rnn_p.in_state_size; k++) {
                d = rnn->rnn_p.prior_weight_ci[j][k] -
                    rnn->rnn_p.weight_ci[j][k];
                post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
            }
            for (int k = 0; k < rnn->rnn_p.c_state_size; k++) {
                d = rnn->rnn_p.prior_weight_cc[j][k] -
                    rnn->rnn_p.weight_cc[j][k];
                post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
            }
            d = rnn->rnn_p.prior_threshold_c[j] - rnn->rnn_p.threshold_c[j];
            post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
            d = rnn->rnn_p.prior_tau[j] - rnn->rnn_p.tau[j];
            post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
        }
        for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
            for (int k = 0; k < rnn->rnn_p.c_state_size; k++) {
                d = rnn->rnn_p.prior_weight_oc[j][k] -
                    rnn->rnn_p.weight_oc[j][k];
                post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
            }
            d = rnn->rnn_p.prior_threshold_o[j] - rnn->rnn_p.threshold_o[j];
            post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
        }
    }
#ifdef ENABLE_ATTRACTION_OF_INIT_C
    for (int i = 0; i < mre->series_num; i++) {
        for (int j = 0; j < mre->expert_num; j++) {
            for (int k = 0; k < mre->expert_rnn[j].rnn_p.c_state_size; k++) {
                double d = mean[i][j][k] -
                    mre->mre_s[i].expert_rnn_s[j]->init_c_inter_state[k];
                post_dist -= (d * d) / (2 * variance[i][j][k]);
                post_dist -= 0.5 * log(2 * M_PI * variance[i][j][k]);
            }
        }
    }
#endif
    return post_dist;
}


#ifdef ENABLE_ATTRACTION_OF_INIT_C
extern void get_mean_and_variance (
        struct rnn_state *rnn_s,
        double *mean,
        double *variance);
#endif


void assert_effect_mre_learn (struct mixture_of_rnn_experts *mre)
{
    double post_dist, next_post_dist;

    mre_forward_backward_dynamics_forall(mre);
#ifdef ENABLE_ATTRACTION_OF_INIT_C
    double ***mean, ***variance;
    MALLOC2(mean, mre->series_num, mre->expert_num);
    MALLOC2(variance, mre->series_num, mre->expert_num);
    for (int i = 0; i < mre->series_num; i++) {
        for (int j = 0; j < mre->expert_num; j++) {
            MALLOC(mean[i][j], mre->expert_rnn[j].rnn_p.c_state_size);
            MALLOC(variance[i][j], mre->expert_rnn[j].rnn_p.c_state_size);
            get_mean_and_variance(mre->mre_s[i].expert_rnn_s[j], mean[i][j],
                    variance[i][j]);
        }
    }
    post_dist = get_posterior_distribution(mre, mean, variance);
#else
    post_dist = get_posterior_distribution(mre);
#endif

    for (int i = 0; i < mre->expert_num; i++) {
        rnn_reset_delta_parameters(&(mre->expert_rnn[i].rnn_p));
    }
    mre_learn(mre, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0.0);

    mre_forward_backward_dynamics_forall(mre);
#ifdef ENABLE_ATTRACTION_OF_INIT_C
    next_post_dist = get_posterior_distribution(mre, mean, variance);
    for (int i = 0; i < mre->series_num; i++) {
        for (int j = 0; j < mre->expert_num; j++) {
            FREE(mean[i][j]);
            FREE(variance[i][j]);
        }
    }
    FREE2(mean);
    FREE2(variance);
#else
    next_post_dist = get_posterior_distribution(mre);
#endif

    mu_assert(post_dist < next_post_dist);
}


void assert_effect_mre_learn_with_adapt_lr (
        struct mixture_of_rnn_experts *mre,
        double *adapt_lr)
{
    double error, next_error;

    mre_forward_backward_dynamics_forall(mre);
    error = 0;
    for (int i = 0; i < mre->series_num; i++) {
        error += mre_get_error( mre->mre_s + i);
    }

    for (int i = 0; i < mre->expert_num; i++) {
        rnn_reset_delta_parameters(&(mre->expert_rnn[i].rnn_p));
    }
    *adapt_lr = mre_learn_with_adapt_lr(mre, *adapt_lr, 1000.0, 1000.0,
            1000.0, 1000.0, 1000.0, 0.0);

    mre_forward_backward_dynamics_forall(mre);
    next_error = 0;
    for (int i = 0; i < mre->series_num; i++) {
        next_error += mre_get_error(mre->mre_s + i);
    }
    mu_assert((next_error / error) < (1.0+1e-9));
}

/* test functions */


static void test_init_mixture_of_rnn_experts (void)
{
    struct mixture_of_rnn_experts mre;
    assert_exit(init_mixture_of_rnn_experts, &mre, 0, 0, 1, 1);
    assert_exit(init_mixture_of_rnn_experts, &mre, 1, -1, 1, 1);
    assert_exit(init_mixture_of_rnn_experts, &mre, 1, 0, 0, 1);
    assert_exit(init_mixture_of_rnn_experts, &mre, 1, 0, 1, 0);
    assert_noexit(init_mixture_of_rnn_experts, &mre, 1, 0, 1, 1);
    free_mixture_of_rnn_experts(&mre);
}

static void test_init_mre_state (void)
{
    struct mre_state mre_s;
    struct mixture_of_rnn_experts mre;

    init_mixture_of_rnn_experts(&mre, 1, 0, 1, 1);

    assert_exit(init_mre_state, &mre_s, &mre, 0);
    assert_noexit(init_mre_state, &mre_s, &mre, 1);

    free_mre_state(&mre_s);
    free_mixture_of_rnn_experts(&mre);
}



static void test_fwrite_mixture_of_rnn_experts (
        struct mixture_of_rnn_experts *mre)
{
    struct mixture_of_rnn_experts mre2;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(mre, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_mixture_of_rnn_experts(&mre2, fp);
    fclose(fp);

    assert_equal_int(mre->expert_num, mre2.expert_num);
    assert_equal_int(mre->series_num, mre2.series_num);
    assert_equal_int(mre->in_state_size, mre2.in_state_size);
    assert_equal_int(mre->out_state_size, mre2.out_state_size);
    assert_equal_int(mre->fixed_gate, mre2.fixed_gate);
    assert_equal_int(mre->gate_prior_distribution,
            mre2.gate_prior_distribution);
    assert_equal_memory(mre->gamma, mre->expert_num, mre2.gamma,
            mre2.expert_num);

    for (int i = 0; i < mre->expert_num; i++) {
        for (int j = 0; j < mre->series_num; j++) {
            assert_equal_pointer(mre->expert_rnn[i].rnn_s + j,
                    mre->mre_s[j].expert_rnn_s[i]);
        }
    }
    for (int i = 0; i < mre2.expert_num; i++) {
        for (int j = 0; j < mre2.series_num; j++) {
            assert_equal_pointer(mre2.expert_rnn[i].rnn_s + j,
                    mre2.mre_s[j].expert_rnn_s[i]);
        }
    }

    for (int i = 0; i < mre->expert_num; i++) {
        assert_equal_rnn_p(&(mre->expert_rnn[i].rnn_p),
                &(mre2.expert_rnn[i].rnn_p));
        for (int j = 0; j < mre->series_num; j++) {
            assert_equal_rnn_s(mre->expert_rnn[i].rnn_s + j,
                    mre2.expert_rnn[i].rnn_s + j);
        }
    }
    for (int i = 0; i < mre->series_num; i++) {
        assert_equal_mre_s(mre->mre_s + i, mre2.mre_s + i);
    }

    free_mixture_of_rnn_experts(&mre2);
}


static void test_mre_get_total_length (
        struct mixture_of_rnn_experts *mre,
        int total_length)
{
    assert_equal_int(total_length, mre_get_total_length(mre));
}


static void test_mre_get_error (struct mixture_of_rnn_experts *mre)
{
    mre_forward_dynamics_forall(mre);
    for (int i = 0; i < mre->series_num; i++) {
        double error = 0;
        for (int n = 0; n < mre->mre_s[i].length; n++) {
            for (int j = 0; j < mre->out_state_size; j++) {
                double d = mre->mre_s[i].out_state[n][j] -
                    mre->mre_s[i].expert_rnn_s[0]->teach_state[n][j];
                error += 0.5 * d * d;
            }
        }
        assert_equal_double(mre_get_error(mre->mre_s + i), error, 10e-10);
    }
}

static void test_mre_get_total_error (struct mixture_of_rnn_experts *mre)
{
    mre_forward_dynamics_forall(mre);
    double total_error = 0;
    for (int i = 0; i < mre->series_num; i++) {
        for (int n = 0; n < mre->mre_s[i].length; n++) {
            for (int j = 0; j < mre->out_state_size; j++) {
                double d = mre->mre_s[i].out_state[n][j] -
                    mre->mre_s[i].expert_rnn_s[0]->teach_state[n][j];
                total_error += 0.5 * d * d;
            }
        }
    }
    assert_equal_double(mre_get_total_error(mre), total_error, 10e-10);
}

static void test_mre_get_joint_likelihood (struct mixture_of_rnn_experts *mre)
{
    mre_forward_dynamics_forall(mre);
    for (int i = 0; i < mre->series_num; i++) {
        double likelihood = 0;
        for (int n = 0; n < mre->mre_s[i].length; n++) {
            double sum = 0;
            for (int j = 0; j < mre->expert_num; j++) {
                struct recurrent_neural_network *rnn = mre->expert_rnn + j;
                double g = 1;
                for (int k = 0; k < mre->out_state_size; k++) {
                    double d = mre->mre_s[i].expert_rnn_s[j]->out_state[n][k] -
                        mre->mre_s[i].expert_rnn_s[j]->teach_state[n][k];
                    g *= (1.0 / sqrt(2 * M_PI * rnn->rnn_p.variance)) *
                        exp(-(d * d) / (2 * rnn->rnn_p.variance));
                }
                sum += mre->mre_s[i].gate[j][n] * g;
            }
            likelihood += log(sum);
        }
        mre_set_likelihood(mre->mre_s + i);
        assert_equal_double(mre_get_joint_likelihood(mre->mre_s + i),
                likelihood, 10e-10);
    }
}

static void test_mre_get_total_joint_likelihood (
        struct mixture_of_rnn_experts *mre)
{
    mre_forward_backward_dynamics_forall(mre);
    double total_likelihood = 0;
    for (int i = 0; i < mre->series_num; i++) {
        for (int n = 0; n < mre->mre_s[i].length; n++) {
            double sum = 0;
            for (int j = 0; j < mre->expert_num; j++) {
                struct recurrent_neural_network *rnn = mre->expert_rnn + j;
                double g = 1;
                for (int k = 0; k < mre->out_state_size; k++) {
                    double d = mre->mre_s[i].expert_rnn_s[j]->out_state[n][k] -
                        mre->mre_s[i].expert_rnn_s[j]->teach_state[n][k];
                    g *= (1.0 / sqrt(2 * M_PI * rnn->rnn_p.variance)) *
                        exp(-(d * d) / (2 * rnn->rnn_p.variance));
                }
                sum += mre->mre_s[i].gate[j][n] * g;
            }
            total_likelihood += log(sum);
        }
    }
    assert_equal_double(mre_get_total_joint_likelihood(mre), total_likelihood,
            10e-10);
}

static void test_mre_get_prior_likelihood (struct mixture_of_rnn_experts *mre)
{
    for (int k = 0; k < 5; k++) {
        for (int i = 0; i < mre->expert_num; i++) {
            mre->gamma[i] = k + 1;
        }
        mre->gate_prior_distribution = NO_DISTRIBUTION;
        mre_forward_dynamics_forall(mre);
        for (int i = 0; i < mre->series_num; i++) {
            double likelihood = 0;
            mre_set_likelihood(mre->mre_s + i);
            assert_equal_double(mre_get_prior_likelihood(mre->mre_s + i),
                    likelihood, 10e-10);
        }

        mre->gate_prior_distribution = GAUSS_DISTRIBUTION;
        mre_forward_dynamics_forall(mre);
        for (int i = 0; i < mre->series_num; i++) {
            double likelihood = 0;
            for (int n = 1; n < mre->mre_s[i].length; n++) {
                for (int j = 0; j < mre->expert_num; j++) {
                    double d = mre->mre_s[i].beta[j][n] -
                        mre->mre_s[i].beta[j][n-1];
                    likelihood -= (d * d) / (2 * mre->gamma[j] * mre->gamma[j]);
                    likelihood -= 0.5 * log(2 * M_PI * mre->gamma[j] *
                            mre->gamma[j]);
                }
            }
            mre_set_likelihood(mre->mre_s + i);
            assert_equal_double(mre_get_prior_likelihood(mre->mre_s + i),
                    likelihood, 10e-10);
        }

        mre->gate_prior_distribution = CAUCHY_DISTRIBUTION;
        mre_forward_dynamics_forall(mre);
        for (int i = 0; i < mre->series_num; i++) {
            double likelihood = 0;
            for (int n = 1; n < mre->mre_s[i].length; n++) {
                for (int j = 0; j < mre->expert_num; j++) {
                    double d = mre->mre_s[i].beta[j][n] -
                        mre->mre_s[i].beta[j][n-1];
                    likelihood += log(mre->gamma[j]) - log(M_PI * (d +
                                mre->gamma[j] * mre->gamma[j]));
                }
            }
            mre_set_likelihood(mre->mre_s + i);
            assert_equal_double(mre_get_prior_likelihood(mre->mre_s + i),
                    likelihood, 10e-10);
        }
    }
}

static void test_mre_get_total_prior_likelihood (
        struct mixture_of_rnn_experts *mre)
{
    for (int k = 0; k < 5; k++) {
        for (int i = 0; i < mre->expert_num; i++) {
            mre->gamma[i] = k + 1;
        }
        mre->gate_prior_distribution = NO_DISTRIBUTION;
        mre_forward_backward_dynamics_forall(mre);
        double total_likelihood = 0;
        assert_equal_double(mre_get_total_prior_likelihood(mre),
                total_likelihood, 10e-10);

        mre->gate_prior_distribution = GAUSS_DISTRIBUTION;
        mre_forward_backward_dynamics_forall(mre);
        total_likelihood = 0;
        for (int i = 0; i < mre->series_num; i++) {
            for (int n = 1; n < mre->mre_s[i].length; n++) {
                for (int j = 0; j < mre->expert_num; j++) {
                    double d = mre->mre_s[i].beta[j][n] -
                        mre->mre_s[i].beta[j][n-1];
                    total_likelihood -= (d * d) /
                        (2 * mre->gamma[j] * mre->gamma[j]);
                    total_likelihood -= 0.5 *
                        log(2 * M_PI * mre->gamma[j] * mre->gamma[j]);
                }
            }
        }
        assert_equal_double(mre_get_total_prior_likelihood(mre),
                total_likelihood, 10e-10);

        mre->gate_prior_distribution = CAUCHY_DISTRIBUTION;
        mre_forward_backward_dynamics_forall(mre);
        total_likelihood = 0;
        for (int i = 0; i < mre->series_num; i++) {
            for (int n = 1; n < mre->mre_s[i].length; n++) {
                for (int j = 0; j < mre->expert_num; j++) {
                    double d = mre->mre_s[i].beta[j][n] -
                        mre->mre_s[i].beta[j][n-1];
                    total_likelihood += log(mre->gamma[j]) -
                        log(M_PI * (d + mre->gamma[j] * mre->gamma[j]));
                }
            }
        }
        assert_equal_double(mre_get_total_prior_likelihood(mre),
                total_likelihood, 10e-10);
    }
}

static void test_mre_clean_target (struct mixture_of_rnn_experts *mre)
{
    mre_clean_target(mre);
    assert_equal_int(0, mre->series_num);
    assert_equal_pointer(NULL, mre->mre_s);
    for (int i = 0; i < mre->expert_num; i++) {
        assert_equal_int(0, mre->expert_rnn[i].series_num);
        assert_equal_pointer(NULL, mre->expert_rnn[i].rnn_s);
    }
}

static void test_mre_forward_dynamics_forall (
        struct mixture_of_rnn_experts *mre)
{
    struct mixture_of_rnn_experts mre2;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(mre, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_mixture_of_rnn_experts(&mre2, fp);
    fclose(fp);

    mre_forward_dynamics_forall(mre);
    struct mre_state *mre_s, *mre2_s;
    for (int i = 0; i < mre->series_num; i++) {
        mre_s = mre->mre_s + i;
        mre2_s = mre2.mre_s + i;
        mre_forward_dynamics(mre2_s);
        assert_equal_vector_sequence(mre2_s->out_state,
                mre2.out_state_size * sizeof(double), mre2_s->length,
                mre_s->out_state, mre->out_state_size * sizeof(double),
                mre_s->length);
    }

    free_mixture_of_rnn_experts(&mre2);
}


static void test_mre_set_likelihood_forall (struct mixture_of_rnn_experts *mre)
{
    struct mixture_of_rnn_experts mre2;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(mre, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_mixture_of_rnn_experts(&mre2, fp);
    fclose(fp);

    mre_forward_dynamics_forall(mre);
    mre_forward_dynamics_forall(&mre2);

    mre_set_likelihood_forall(mre);
    struct mre_state *mre_s, *mre2_s;
    struct rnn_state *rnn_s, *rnn2_s;
    for (int i = 0; i < mre->series_num; i++) {
        mre_s = mre->mre_s + i;
        mre2_s = mre2.mre_s + i;
        mre_set_likelihood(mre2_s);

        assert_equal_memory(mre_s->joint_likelihood,
                mre_s->length * sizeof(double), mre2_s->joint_likelihood,
                mre2_s->length * sizeof(double));
        assert_equal_vector_sequence(mre2_s->generation_likelihood,
                mre2_s->length * sizeof(double), mre2_s->mre->expert_num,
                mre_s->generation_likelihood, mre_s->length * sizeof(double),
            mre_s->mre->expert_num);
        assert_equal_vector_sequence(mre2_s->discrimination_likelihood,
                mre2_s->length * sizeof(double), mre2_s->mre->expert_num,
                mre_s->discrimination_likelihood,
                mre_s->length * sizeof(double), mre_s->mre->expert_num);
        assert_equal_vector_sequence(mre2_s->prior_likelihood,
                mre2_s->length * sizeof(double), mre2_s->mre->expert_num,
                mre_s->prior_likelihood, mre_s->length * sizeof(double),
                mre_s->mre->expert_num);

        for (int j = 0; j < mre->expert_num; j++) {
            rnn_s = mre_s->expert_rnn_s[j];
            rnn2_s = mre2_s->expert_rnn_s[j];
            assert_equal_vector_sequence(rnn2_s->likelihood,
                    rnn2_s->rnn_p->out_state_size * sizeof(double),
                    rnn2_s->length, rnn_s->likelihood,
                    rnn_s->rnn_p->out_state_size * sizeof(double),
                    rnn_s->length);
            assert_equal_vector_sequence(rnn2_s->delta_likelihood,
                    rnn2_s->rnn_p->out_state_size * sizeof(double),
                    rnn2_s->length, rnn_s->delta_likelihood,
                    rnn_s->rnn_p->out_state_size * sizeof(double),
                    rnn_s->length);
        }
    }

    free_mixture_of_rnn_experts(&mre2);
}


static void test_mre_backward_dynamics_forall (
        struct mixture_of_rnn_experts *mre)
{
    struct mixture_of_rnn_experts mre2;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(mre, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_mixture_of_rnn_experts(&mre2, fp);
    fclose(fp);

    mre_forward_dynamics_forall(mre);
    mre_set_likelihood_forall(mre);
    mre_forward_dynamics_forall(&mre2);
    mre_set_likelihood_forall(&mre2);

    mre_backward_dynamics_forall(mre);
    struct mre_state *mre_s, *mre2_s;
    struct rnn_state *rnn_s, *rnn2_s;
    for (int i = 0; i < mre->series_num; i++) {
        mre_s = mre->mre_s + i;
        mre2_s = mre2.mre_s + i;
        mre_backward_dynamics(mre2_s);
        for (int j = 0; j < mre->expert_num; j++) {
            rnn_s = mre_s->expert_rnn_s[j];
            rnn2_s = mre2_s->expert_rnn_s[j];
            assert_equal_double(rnn_s->delta_s, rnn2_s->delta_s, 0.0);
            assert_equal_vector_sequence(rnn2_s->delta_c_inter,
                    rnn2_s->rnn_p->c_state_size * sizeof(double),
                    rnn2_s->length, rnn_s->delta_c_inter,
                    rnn_s->rnn_p->c_state_size * sizeof(double),
                    rnn_s->length);
            assert_equal_vector_sequence(rnn2_s->delta_o_inter,
                    rnn2_s->rnn_p->out_state_size * sizeof(double),
                    rnn2_s->length, rnn_s->delta_o_inter,
                    rnn_s->rnn_p->out_state_size * sizeof(double),
                    rnn_s->length);
        }
    }

    free_mixture_of_rnn_experts(&mre2);
}

static void test_mre_forward_dynamics_in_closed_loop_forall (
        struct mixture_of_rnn_experts *mre)
{
    struct mixture_of_rnn_experts mre2;
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
    fclose(fp);

    mre_forward_dynamics_in_closed_loop_forall(mre, 1);
    struct mre_state *mre_s, *mre2_s;
    for (int i = 0; i < mre->series_num; i++) {
        mre_s = mre->mre_s + i;
        mre2_s = mre2.mre_s + i;
        mre_forward_dynamics_in_closed_loop(mre2_s, 1);
        assert_equal_vector_sequence(mre2_s->out_state,
                mre2.out_state_size * sizeof(double), mre2_s->length,
                mre_s->out_state, mre->out_state_size * sizeof(double),
                mre_s->length);
    }

    free_mixture_of_rnn_experts(&mre2);
}


static void test_mre_learn (struct mixture_of_rnn_experts *mre)
{
    for (int n = 0; n < 3; n++) {
        for (int i = 0; i < mre->expert_num; i++) {
            mre->expert_rnn[i].rnn_p.prior_strength = 0.01 * n;
        }
        for (int i = 0; i < 6; i++) {
            mre->fixed_gate = (i==0)?0:1;
            for (int j = 0; j < mre->expert_num; j++) {
                mre->expert_rnn[j].rnn_p.fixed_weight = (i==1)?0:1;
                mre->expert_rnn[j].rnn_p.fixed_threshold = (i==2)?0:1;
                mre->expert_rnn[j].rnn_p.fixed_tau = (i==3)?0:1;
                mre->expert_rnn[j].rnn_p.fixed_init_c_state = (i==4)?0:1;
                mre->expert_rnn[j].rnn_p.fixed_sigma = (i==5)?0:1;
            }
            if (i==0) {
                mre->gate_prior_distribution = NO_DISTRIBUTION;
                assert_effect_mre_learn(mre);
                mre->gate_prior_distribution = GAUSS_DISTRIBUTION;
                assert_effect_mre_learn(mre);
                mre->gate_prior_distribution = CAUCHY_DISTRIBUTION;
                assert_effect_mre_learn(mre);
            } else {
                mre->gate_prior_distribution = NO_DISTRIBUTION;
                assert_effect_mre_learn(mre);
            }
        }

        mre->fixed_gate = 0;
        for (int i = 0; i < mre->expert_num; i++) {
            mre->expert_rnn[i].rnn_p.fixed_weight = 0;
            mre->expert_rnn[i].rnn_p.fixed_threshold = 0;
            mre->expert_rnn[i].rnn_p.fixed_tau = 0;
            mre->expert_rnn[i].rnn_p.fixed_init_c_state = 0;
            mre->expert_rnn[i].rnn_p.fixed_sigma = 0;
        }
        mre->gate_prior_distribution = NO_DISTRIBUTION;
        assert_effect_mre_learn(mre);
        mre->gate_prior_distribution = GAUSS_DISTRIBUTION;
        assert_effect_mre_learn(mre);
        mre->gate_prior_distribution = CAUCHY_DISTRIBUTION;
        assert_effect_mre_learn(mre);
    }
}


static void test_mre_learn_s (struct mixture_of_rnn_experts *mre)
{
    struct mixture_of_rnn_experts mre2;
    FILE *fp;

    mre->fixed_gate = 0;
    for (int i = 0; i < mre->expert_num; i++) {
        mre->expert_rnn[i].rnn_p.fixed_weight = 0;
        mre->expert_rnn[i].rnn_p.fixed_threshold = 0;
        mre->expert_rnn[i].rnn_p.fixed_tau = 0;
        mre->expert_rnn[i].rnn_p.fixed_init_c_state = 0;
        mre->expert_rnn[i].rnn_p.fixed_sigma = 0;
    }

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(mre, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_mixture_of_rnn_experts(&mre2, fp);
    fclose(fp);

    int total_length = mre_get_total_length(mre);
    double rho = 1e-8;
    double rho_gate = rho;
    double rho_weight = rho / (total_length * mre->out_state_size);
    double rho_tau = rho / (total_length * mre->out_state_size);
    double rho_sigma = rho / (total_length * mre->out_state_size);
    double rho_init = rho / mre->out_state_size;
    double adapt_lr = 1.0;


    for (int n = 0; n < 6; n++) {
        if (n % 3 == 0) {
            mre->gate_prior_distribution = NO_DISTRIBUTION;
            mre2.gate_prior_distribution = NO_DISTRIBUTION;
        } else if (n % 3 == 1) {
            mre->gate_prior_distribution = GAUSS_DISTRIBUTION;
            mre2.gate_prior_distribution = GAUSS_DISTRIBUTION;
        } else {
            mre->gate_prior_distribution = CAUCHY_DISTRIBUTION;
            mre2.gate_prior_distribution = CAUCHY_DISTRIBUTION;
        }
        if (n < 3) {
            mre_learn(mre, rho_gate, rho_weight, rho_tau, rho_init,
                    rho_sigma, 0);
            mre_learn_s(&mre2, rho, 0);
        } else {
            mre_learn_with_adapt_lr(mre, adapt_lr, rho_gate, rho_weight,
                    rho_tau, rho_init, rho_sigma, 0);
            mre_learn_s_with_adapt_lr(&mre2, adapt_lr, rho, 0);
        }
        for (int i = 0; i < mre->expert_num; i++) {
            assert_equal_rnn_p(&(mre->expert_rnn[i].rnn_p),
                    &(mre2.expert_rnn[i].rnn_p));
            for (int j = 0; j < mre->series_num; j++) {
                assert_equal_rnn_s(mre->expert_rnn[i].rnn_s + j,
                        mre2.expert_rnn[i].rnn_s + j);
            }
        }
        for (int i = 0; i < mre->series_num; i++) {
            assert_equal_mre_s(mre->mre_s + i, mre2.mre_s + i);
        }
    }

    free_mixture_of_rnn_experts(&mre2);
}

static void test_mre_backup_learning_parameters (
        struct mixture_of_rnn_experts *mre)
{
    struct mixture_of_rnn_experts tmp_mre;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_mixture_of_rnn_experts(mre, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_mixture_of_rnn_experts(&tmp_mre, fp);
    fclose(fp);


    mre_backup_learning_parameters(mre);

    for (int i = 0; i < mre->expert_num; i++) {
        struct recurrent_neural_network *rnn;
        size_t in_msz, c_msz, out_msz;
        rnn = mre->expert_rnn + i;
        in_msz = rnn->rnn_p.in_state_size * sizeof(double);
        c_msz = rnn->rnn_p.c_state_size * sizeof(double);
        out_msz = rnn->rnn_p.out_state_size * sizeof(double);
        rnn->rnn_p.sigma = 0;
        rnn->rnn_p.variance = 1;
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            memset(rnn->rnn_p.weight_ci[j], 0, in_msz);
            memset(rnn->rnn_p.weight_cc[j], 0, c_msz);
        }
        for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
            memset(rnn->rnn_p.weight_oc[j], 0, c_msz);
        }
        memset(rnn->rnn_p.threshold_c, 0, c_msz);
        memset(rnn->rnn_p.threshold_o, 0, out_msz);
        memset(rnn->rnn_p.tau, 0, c_msz);
        memset(rnn->rnn_p.eta, 0, c_msz);
        for (int j = 0; j < rnn->series_num; j++) {
            memset(rnn->rnn_s[j].init_c_inter_state, 0, c_msz);
            memset(rnn->rnn_s[j].init_c_state, 0, c_msz);
        }
    }
    for (int i = 0; i < mre->series_num; i++) {
        size_t len_msz;
        len_msz = mre->mre_s[i].length * sizeof(double);
        for (int j = 0; j < mre->expert_num; j++) {
            memset(mre->mre_s[i].gate[j], 0, len_msz);
            memset(mre->mre_s[i].beta[j], 0, len_msz);
        }
    }
    mre_restore_learning_parameters(mre);

    for (int i = 0; i < mre->expert_num; i++) {
        struct recurrent_neural_network *rnn, *tmp_rnn;
        size_t in_msz, c_msz, out_msz;
        rnn = mre->expert_rnn + i;
        tmp_rnn = tmp_mre.expert_rnn + i;
        in_msz = rnn->rnn_p.in_state_size * sizeof(double);
        c_msz = rnn->rnn_p.c_state_size * sizeof(double);
        out_msz = rnn->rnn_p.out_state_size * sizeof(double);
        assert_equal_double(tmp_rnn->rnn_p.sigma, rnn->rnn_p.sigma, 0.0);
        assert_equal_double(tmp_rnn->rnn_p.variance, rnn->rnn_p.variance, 0.0);
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            assert_equal_memory(tmp_rnn->rnn_p.weight_ci[j], in_msz,
                    rnn->rnn_p.weight_ci[j], in_msz);
            assert_equal_memory(tmp_rnn->rnn_p.weight_cc[j], c_msz,
                    rnn->rnn_p.weight_cc[j], c_msz);
        }
        for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
            assert_equal_memory(tmp_rnn->rnn_p.weight_oc[j], c_msz,
                    rnn->rnn_p.weight_oc[j], c_msz);
        }
        assert_equal_memory(tmp_rnn->rnn_p.threshold_c, c_msz,
                rnn->rnn_p.threshold_c, c_msz);
        assert_equal_memory(tmp_rnn->rnn_p.threshold_o, out_msz,
                rnn->rnn_p.threshold_o, out_msz);
        assert_equal_memory(tmp_rnn->rnn_p.tau, c_msz, rnn->rnn_p.tau,
                c_msz);
        assert_equal_memory(tmp_rnn->rnn_p.eta, c_msz, rnn->rnn_p.eta,
                c_msz);
        for (int j = 0; j < rnn->series_num; j++) {
            assert_equal_memory(tmp_rnn->rnn_s[j].init_c_inter_state, c_msz,
                    rnn->rnn_s[j].init_c_inter_state, c_msz);
            assert_equal_memory(tmp_rnn->rnn_s[j].init_c_state, c_msz,
                    rnn->rnn_s[j].init_c_state, c_msz);
        }
    }
    for (int i = 0; i < mre->series_num; i++) {
        size_t len_msz;
        len_msz = mre->mre_s[i].length * sizeof(double);
        for (int j = 0; j < mre->expert_num; j++) {
            assert_equal_memory(tmp_mre.mre_s[i].gate[j], len_msz,
                    mre->mre_s[i].gate[j], len_msz);
            assert_equal_memory(tmp_mre.mre_s[i].beta[j], len_msz,
                    mre->mre_s[i].beta[j], len_msz);
        }
    }

    free_mixture_of_rnn_experts(&tmp_mre);
}


static void test_mre_learn_with_adapt_lr (
        struct mixture_of_rnn_experts *mre)
{
    double adapt_lr = 1.0;
    for (int i = 0; i < mre->expert_num; i++) {
        mre->expert_rnn[i].rnn_p.prior_strength = 0;
    }
    for (int i = 0; i < 6; i++) {
        mre->fixed_gate = (i==0)?0:1;
        for (int j = 0; j < mre->expert_num; j++) {
            mre->expert_rnn[j].rnn_p.fixed_weight = (i==1)?0:1;
            mre->expert_rnn[j].rnn_p.fixed_threshold = (i==2)?0:1;
            mre->expert_rnn[j].rnn_p.fixed_tau = (i==3)?0:1;
            mre->expert_rnn[j].rnn_p.fixed_init_c_state = (i==4)?0:1;
            mre->expert_rnn[j].rnn_p.fixed_sigma = (i==5)?0:1;
        }
        if (i==0) {
            mre->gate_prior_distribution = NO_DISTRIBUTION;
            assert_effect_mre_learn_with_adapt_lr(mre, &adapt_lr);
            mre->gate_prior_distribution = GAUSS_DISTRIBUTION;
            assert_effect_mre_learn_with_adapt_lr(mre, &adapt_lr);
            mre->gate_prior_distribution = CAUCHY_DISTRIBUTION;
            assert_effect_mre_learn_with_adapt_lr(mre, &adapt_lr);
        } else {
            mre->gate_prior_distribution = NO_DISTRIBUTION;
            assert_effect_mre_learn_with_adapt_lr(mre, &adapt_lr);
        }
    }

    mre->fixed_gate = 0;
    for (int i = 0; i < mre->expert_num; i++) {
        mre->expert_rnn[i].rnn_p.fixed_weight = 0;
        mre->expert_rnn[i].rnn_p.fixed_threshold = 0;
        mre->expert_rnn[i].rnn_p.fixed_tau = 0;
        mre->expert_rnn[i].rnn_p.fixed_init_c_state = 0;
        mre->expert_rnn[i].rnn_p.fixed_sigma = 0;
    }
    mre->gate_prior_distribution = NO_DISTRIBUTION;
    assert_effect_mre_learn_with_adapt_lr(mre, &adapt_lr);
    mre->gate_prior_distribution = GAUSS_DISTRIBUTION;
    assert_effect_mre_learn_with_adapt_lr(mre, &adapt_lr);
    mre->gate_prior_distribution = CAUCHY_DISTRIBUTION;
    assert_effect_mre_learn_with_adapt_lr(mre, &adapt_lr);
}


static void test_mre_update_prior_strength (
        struct mixture_of_rnn_experts *mre)
{
    double tmp_value[mre->expert_num];
    for (int i = 0; i < mre->expert_num; i++) {
        tmp_value[i] = mre->expert_rnn[i].rnn_p.prior_strength;
    }
    double lambda, alpha, likelihood, value;
    lambda = 0.9;
    alpha = 1.0;
    mre_update_prior_strength(mre, lambda, alpha);
    for (int i = 0; i < mre->expert_num; i++) {
        likelihood = 0;
        for (int j = 0; j < mre->series_num; j++) {
            for (int n = 0; n < mre->mre_s[j].length; n++) {
                likelihood += mre->mre_s[j].discrimination_likelihood[i][n] *
                    mre->mre_s[j].joint_likelihood[n];
            }
        }
        value = lambda * tmp_value[i] + alpha * likelihood;
        assert_equal_double(value,
                mre->expert_rnn[i].rnn_p.prior_strength, 1e-10);
    }
}




void test_mre_state_setup (
        struct mixture_of_rnn_experts *mre,
        int target_num,
        int *target_length)
{
    const int in_state_size = mre->in_state_size;
    const int out_state_size = mre->out_state_size;
    for (int i = 0; i < target_num; i++) {
        double **input, **target;
        MALLOC2(input, target_length[i], in_state_size);
        MALLOC2(target, target_length[i], out_state_size);
        for (int n = 0; n < target_length[i]; n++) {
            for (int j = 0; j < in_state_size; j++) {
                input[n][j] = genrand_real1();
            }
            for (int j = 0; j < out_state_size; j++) {
                target[n][j] = genrand_real1();
            }
        }
        mre_add_target(mre, target_length[i], (const double* const*)input,
                (const double* const*)target);
        FREE2(input);
        FREE2(target);
    }
}


typedef struct test_mre_data {
    struct mixture_of_rnn_experts mre;
    int target_num;
    int total_length;
} test_mre_data;

static void test_mre_data_setup (
        struct test_mre_data *t_data,
        unsigned long seed,
        int expert_num,
        int in_state_size,
        int c_state_size,
        int out_state_size,
        int target_num,
        int *target_length)
{
    struct mixture_of_rnn_experts *mre = &t_data->mre;

    init_genrand(seed);

    init_mixture_of_rnn_experts(mre, expert_num, in_state_size, c_state_size,
            out_state_size);

    t_data->target_num = target_num;
    t_data->total_length = 0;
    for (int i = 0; i < target_num; i++) {
        t_data->total_length += target_length[i];
    }
    test_mre_state_setup(mre, target_num, target_length);
}


void test_mre (void)
{
    mu_run_test(test_init_mixture_of_rnn_experts);
    mu_run_test(test_init_mre_state);

    struct test_mre_data t_data[3];
    test_mre_data_setup(t_data, 3837L, 8, 10, 15, 7, 3, (int[]){100,100,50});
    test_mre_data_setup(t_data+1, 937112L, 4, 4, 9, 4, 2, (int[]){50,100});
    test_mre_data_setup(t_data+2, 1112L, 8, 0, 10, 1, 4,
            (int[]){75,60,100,125});
    for (int i = 0; i < 3; i++) {
        mu_run_test_with_args(test_fwrite_mixture_of_rnn_experts,
                &t_data[i].mre);
        mu_run_test_with_args(test_mre_get_total_length, &t_data[i].mre,
                t_data[i].total_length);
        mu_run_test_with_args(test_mre_get_error, &t_data[i].mre);
        mu_run_test_with_args(test_mre_get_total_error, &t_data[i].mre);
        mu_run_test_with_args(test_mre_get_joint_likelihood, &t_data[i].mre);
        mu_run_test_with_args(test_mre_get_total_joint_likelihood,
                &t_data[i].mre);
        mu_run_test_with_args(test_mre_get_prior_likelihood, &t_data[i].mre);
        mu_run_test_with_args(test_mre_get_total_prior_likelihood,
                &t_data[i].mre);
        mu_run_test_with_args(test_mre_forward_dynamics_forall, &t_data[i].mre);
        mu_run_test_with_args(test_mre_set_likelihood_forall, &t_data[i].mre);
        mu_run_test_with_args(test_mre_backward_dynamics_forall,
                &t_data[i].mre);
        mu_run_test_with_args(test_mre_forward_dynamics_in_closed_loop_forall,
                &t_data[i].mre);
        mu_run_test_with_args(test_mre_learn, &t_data[i].mre);
        mu_run_test_with_args(test_mre_learn_s, &t_data[i].mre);
        mu_run_test_with_args(test_mre_backup_learning_parameters,
                &t_data[i].mre);
        mu_run_test_with_args(test_mre_learn_with_adapt_lr, &t_data[i].mre);
        mu_run_test_with_args(test_mre_update_prior_strength, &t_data[i].mre);
        mu_run_test_with_args(test_mre_clean_target, &t_data[i].mre);
    }
    for (int i = 0; i < 3; i++) {
        free_mixture_of_rnn_experts(&t_data[i].mre);
    }
}


