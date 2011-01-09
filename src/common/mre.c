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
#include <assert.h>
#include "mt19937ar.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "mre.h"


#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif


#ifndef FIXED_GATE
#define FIXED_GATE 0
#endif

#ifndef INIT_GAMMA
#define INIT_GAMMA 10
#endif


#ifdef ENABLE_ADAPTIVE_LEARNING_RATE

#ifndef MAX_ITERATION_IN_ADAPTIVE_LR
#define MAX_ITERATION_IN_ADAPTIVE_LR 1000
#endif

#ifndef MAX_PERF_INC
#define MAX_PERF_INC 1.1
#endif

#ifndef LR_DEC
#define LR_DEC 0.7
#endif

#ifndef LR_INC
#define LR_INC 1.05
#endif

#endif // ENABLE_ADAPTIVE_LEARNING_RATE


/******************************************************************************/
/********** Initialization and Free *******************************************/
/******************************************************************************/


void init_mre_state (
        struct mre_state *mre_s,
        struct mixture_of_rnn_experts *mre,
        int length)
{
    assert(length > 0);

    mre_s->mre = mre;
    mre_s->length = length;

    mre_state_alloc(mre_s);

    for (int i = 0; i < mre->expert_num; i++) {
        for (int n = 0; n < length; n++) {
            mre_s->gate[i][n] = 1.0/(double)mre->expert_num;
            mre_s->beta[i][n] = 0;
            mre_s->delta_beta[i][n] = 0;
        }
        mre_s->expert_rnn_s[i] = NULL;
    }
}


void init_mixture_of_rnn_experts (
        struct mixture_of_rnn_experts *mre,
        int expert_num,
        int in_state_size,
        int c_state_size,
        int out_state_size)
{
    /*
     * Mixture of RNN experts has to contain at least one expert RNN.
     * Each RNN expert has to contain at least one context neuron and one output
     * neuron.
     * An input neuron is not necessarily required.
     */
    assert(expert_num >= 1);
    assert(in_state_size >= 0);
    assert(c_state_size >= 1);
    assert(out_state_size >= 1);

    mre->expert_num = expert_num;
    mre->series_num = 0;
    mre->in_state_size = in_state_size;
    mre->out_state_size = out_state_size;
    mre->fixed_gate = FIXED_GATE;
    mre->gate_prior_distribution = GAUSS_DISTRIBUTION;
    mre->mre_s = NULL;

    MALLOC(mre->gamma, expert_num);
    MALLOC(mre->expert_rnn, expert_num);
    for (int i = 0; i < expert_num; i++) {
        mre->gamma[i] = INIT_GAMMA;
        init_recurrent_neural_network(mre->expert_rnn + i,
                in_state_size, c_state_size, out_state_size);
    }
}


void mre_add_target (
        struct mixture_of_rnn_experts *mre,
        int length,
        double **input,
        double **target)
{
    mre->series_num++;
    REALLOC(mre->mre_s, mre->series_num);
    init_mre_state(mre->mre_s + (mre->series_num-1), mre, length);
    for (int i = 0; i < mre->expert_num; i++) {
        rnn_add_target(mre->expert_rnn + i, length, input, target);
        for (int j = 0; j < mre->series_num; j++) {
            mre->mre_s[j].expert_rnn_s[i] = mre->expert_rnn[i].rnn_s + j;
        }
    }
}


void mre_clean_target (struct mixture_of_rnn_experts *mre)
{
    for (int i = 0; i < mre->expert_num; i++) {
        rnn_clean_target(mre->expert_rnn + i);
    }
    for (int i = 0; i < mre->series_num; i++) {
        free_mre_state(mre->mre_s + i);
    }
    free(mre->mre_s);
    mre->mre_s = NULL;
    mre->series_num = 0;
}


void mre_state_alloc (struct mre_state *mre_s)
{
    const int expert_num = mre_s->mre->expert_num;
    const int out_state_size = mre_s->mre->out_state_size;
    const int length = mre_s->length;

    MALLOC(mre_s->joint_likelihood, length);

    MALLOC(mre_s->gate, expert_num);
    MALLOC(mre_s->beta, expert_num);
    MALLOC(mre_s->delta_beta, expert_num);
    MALLOC(mre_s->generation_likelihood, expert_num);
    MALLOC(mre_s->discrimination_likelihood, expert_num);
    MALLOC(mre_s->prior_likelihood, expert_num);
    MALLOC(mre_s->gate[0], length * expert_num);
    MALLOC(mre_s->beta[0], length * expert_num);
    MALLOC(mre_s->delta_beta[0], length * expert_num);
    MALLOC(mre_s->generation_likelihood[0], length * expert_num);
    MALLOC(mre_s->discrimination_likelihood[0], length * expert_num);
    MALLOC(mre_s->prior_likelihood[0], length * expert_num);
    for (int i = 0; i < expert_num; i++) {
        mre_s->gate[i] = mre_s->gate[0] + (i * length);
        mre_s->beta[i] = mre_s->beta[0] + (i * length);
        mre_s->delta_beta[i] = mre_s->delta_beta[0] + (i * length);
        mre_s->generation_likelihood[i] = mre_s->generation_likelihood[0] +
            (i * length);
        mre_s->discrimination_likelihood[i] =
            mre_s->discrimination_likelihood[0] + (i * length);
        mre_s->prior_likelihood[i] = mre_s->prior_likelihood[0] + (i * length);
    }

    MALLOC(mre_s->out_state, length);
    MALLOC(mre_s->out_state[0], out_state_size * length);
    for (int i = 0; i < length; i++) {
        mre_s->out_state[i] = mre_s->out_state[0] + (i * out_state_size);
    }
    MALLOC(mre_s->expert_rnn_s, expert_num);

#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    MALLOC(mre_s->tmp_gate, length * expert_num);
    MALLOC(mre_s->tmp_beta, length * expert_num);
#endif
}


void free_mre_state (struct mre_state *mre_s)
{
    free(mre_s->joint_likelihood);
    free(mre_s->gate[0]);
    free(mre_s->beta[0]);
    free(mre_s->delta_beta[0]);
    free(mre_s->generation_likelihood[0]);
    free(mre_s->discrimination_likelihood[0]);
    free(mre_s->prior_likelihood[0]);
    free(mre_s->out_state[0]);
    free(mre_s->gate);
    free(mre_s->beta);
    free(mre_s->delta_beta);
    free(mre_s->generation_likelihood);
    free(mre_s->discrimination_likelihood);
    free(mre_s->prior_likelihood);
    free(mre_s->out_state);
    free(mre_s->expert_rnn_s);
#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    free(mre_s->tmp_gate);
    free(mre_s->tmp_beta);
#endif
}


void free_mixture_of_rnn_experts (struct mixture_of_rnn_experts *mre)
{
    free(mre->gamma);
    for (int i = 0; i < mre->expert_num; i++) {
        free_recurrent_neural_network(mre->expert_rnn + i);
    }
    free(mre->expert_rnn);
    for (int i = 0; i < mre->series_num; i++) {
        free_mre_state(mre->mre_s + i);
    }
    free(mre->mre_s);
    mre->mre_s = NULL;
    mre->series_num = 0;
}


/******************************************************************************/
/********** File IO ***********************************************************/
/******************************************************************************/

void fwrite_mre_state (
        const struct mre_state *mre_s,
        FILE *fp)
{
    FWRITE(&mre_s->length, 1, fp);

    for (int i = 0; i < mre_s->mre->expert_num; i++) {
        FWRITE(mre_s->gate[i], mre_s->length, fp);
        FWRITE(mre_s->beta[i], mre_s->length, fp);
        FWRITE(mre_s->delta_beta[i], mre_s->length, fp);
    }
}

void fread_mre_state (
        struct mre_state *mre_s,
        FILE *fp)
{
    FREAD(&mre_s->length, 1, fp);

    mre_state_alloc(mre_s);

    for (int i = 0; i < mre_s->mre->expert_num; i++) {
        FREAD(mre_s->gate[i], mre_s->length, fp);
        FREAD(mre_s->beta[i], mre_s->length, fp);
        FREAD(mre_s->delta_beta[i], mre_s->length, fp);
    }
}



void fwrite_mixture_of_rnn_experts (
        const struct mixture_of_rnn_experts *mre,
        FILE *fp)
{
    FWRITE(&mre->expert_num, 1, fp);
    FWRITE(&mre->series_num, 1, fp);
    FWRITE(&mre->in_state_size, 1, fp);
    FWRITE(&mre->out_state_size, 1, fp);
    FWRITE(&mre->fixed_gate, 1, fp);
    FWRITE(&mre->gate_prior_distribution, 1, fp);

    FWRITE(mre->gamma, mre->expert_num, fp);
    for (int i = 0; i < mre->expert_num; i++) {
        fwrite_recurrent_neural_network(mre->expert_rnn + i, fp);
    }
    for (int i = 0; i < mre->series_num; i++) {
        fwrite_mre_state(mre->mre_s + i, fp);
    }
}


void fread_mixture_of_rnn_experts (
        struct mixture_of_rnn_experts *mre,
        FILE *fp)
{
    FREAD(&mre->expert_num, 1, fp);
    FREAD(&mre->series_num, 1, fp);
    FREAD(&mre->in_state_size, 1, fp);
    FREAD(&mre->out_state_size, 1, fp);
    FREAD(&mre->fixed_gate, 1, fp);
    FREAD(&mre->gate_prior_distribution, 1, fp);

    MALLOC(mre->gamma, mre->expert_num);
    MALLOC(mre->expert_rnn, mre->expert_num);
    MALLOC(mre->mre_s, mre->series_num);

    FREAD(mre->gamma, mre->expert_num, fp);
    for (int i = 0; i < mre->expert_num; i++) {
        fread_recurrent_neural_network(mre->expert_rnn + i, fp);
    }
    for (int i = 0; i < mre->series_num; i++) {
        mre->mre_s[i].mre = mre;
        fread_mre_state(mre->mre_s + i, fp);
        for (int j = 0; j < mre->expert_num; j++) {
            mre->mre_s[i].expert_rnn_s[j] = mre->expert_rnn[j].rnn_s + i;
        }
    }
}




/******************************************************************************/
/********** Computation of Mixture-of-RNN-Experts *****************************/
/******************************************************************************/


int mre_get_total_length (const struct mixture_of_rnn_experts *mre)
{
    int total_length = 0;
    for (int i = 0; i < mre->series_num; i++) {
        total_length += mre->mre_s[i].length;
    }
    return total_length;
}


double mre_get_error (const struct mre_state *mre_s)
{
    double error = 0;
    for (int n = 0; n < mre_s->length; n++) {
        for (int i = 0; i < mre_s->mre->out_state_size; i++) {
            double d = mre_s->out_state[n][i] -
                mre_s->expert_rnn_s[0]->teach_state[n][i];
            error += 0.5 * d * d;
        }
    }
    return error;
}

double mre_get_total_error (const struct mixture_of_rnn_experts *mre)
{
    double error[mre->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        error[i] = mre_get_error(mre->mre_s + i);
    }
    double total_error = 0;
    for (int i = 0; i < mre->series_num; i++) {
        total_error += error[i];
    }
    return total_error;
}

double mre_get_joint_likelihood (const struct mre_state *mre_s)
{
    double likelihood;
    likelihood = 0;
    for (int n = 0; n < mre_s->length; n++) {
        likelihood += log(mre_s->joint_likelihood[n]);
    }
    return likelihood;
}

double mre_get_total_joint_likelihood (const struct mixture_of_rnn_experts *mre)
{
    double likelihood[mre->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        likelihood[i] = mre_get_joint_likelihood(mre->mre_s + i);
    }
    double total_likelihood = 0;
    for (int i = 0; i < mre->series_num; i++) {
        total_likelihood += likelihood[i];
    }
    return total_likelihood;
}

double mre_get_prior_likelihood (const struct mre_state *mre_s)
{
    double likelihood;
    likelihood = 0;
    for (int n = 0; n < mre_s->length; n++) {
        for (int i = 0; i < mre_s->mre->expert_num; i++) {
            likelihood += log(mre_s->prior_likelihood[i][n]);
        }
    }
    return likelihood;
}

double mre_get_total_prior_likelihood (const struct mixture_of_rnn_experts *mre)
{
    double likelihood[mre->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        likelihood[i] = mre_get_prior_likelihood(mre->mre_s + i);
    }
    double total_likelihood = 0;
    for (int i = 0; i < mre->series_num; i++) {
        total_likelihood += likelihood[i];
    }
    return total_likelihood;
}


static void set_out_state_from_expert_rnns (struct mre_state *mre_s)
{
    const int expert_num = mre_s->mre->expert_num;
    const int length = mre_s->length;
    const int out_state_size = mre_s->mre->out_state_size;
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < out_state_size; i++) {
            mre_s->out_state[n][i] = 0;
            for (int j = 0; j < expert_num; j++) {
                mre_s->out_state[n][i] += mre_s->gate[j][n] *
                    mre_s->expert_rnn_s[j]->out_state[n][i];
            }
        }
    }
}


void mre_forward_dynamics (struct mre_state *mre_s)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre_s->mre->expert_num; i++) {
        rnn_forward_dynamics(mre_s->expert_rnn_s[i]);
    }
    set_out_state_from_expert_rnns(mre_s);
}


void mre_forward_dynamics_forall (struct mixture_of_rnn_experts *mre)
{
#ifdef _OPENMP
    const int total_num = mre->series_num * mre->expert_num;
#pragma omp parallel
    {
#pragma omp for
        for (int n = 0; n < total_num; n++) {
            int i = n / mre->expert_num;
            int j = n % mre->expert_num;
            rnn_forward_dynamics(mre->mre_s[i].expert_rnn_s[j]);
        }
#pragma omp for
        for (int i = 0; i < mre->series_num; i++) {
            set_out_state_from_expert_rnns(mre->mre_s + i);
        }
    }
#else
    for (int i = 0; i < mre->series_num; i++) {
        mre_forward_dynamics(mre->mre_s + i);
    }
#endif
}



void mre_forward_dynamics_in_closed_loop (
        struct mre_state *mre_s,
        int delay_length)
{
    struct rnn_state *rnn_s;
    assert(mre_s->mre->in_state_size <= mre_s->mre->out_state_size);
    for (int n = 0; n < mre_s->length; n++) {
        if (n == 0) {
            for (int i = 0; i < mre_s->mre->expert_num; i++) {
                rnn_s = mre_s->expert_rnn_s[i];
                rnn_forward_map(rnn_s->rnn_p, rnn_s->in_state[0],
                        rnn_s->init_c_inter_state, rnn_s->init_c_state,
                        rnn_s->c_inputsum[0], rnn_s->c_inter_state[0],
                        rnn_s->c_state[0], rnn_s->o_inter_state[0],
                        rnn_s->out_state[0]);
            }
        } else if (n < delay_length) {
            for (int i = 0; i < mre_s->mre->expert_num; i++) {
                rnn_s = mre_s->expert_rnn_s[i];
                rnn_forward_map(rnn_s->rnn_p, rnn_s->in_state[n],
                        rnn_s->c_inter_state[n-1], rnn_s->c_state[n-1],
                        rnn_s->c_inputsum[n], rnn_s->c_inter_state[n],
                        rnn_s->c_state[n], rnn_s->o_inter_state[n],
                        rnn_s->out_state[n]);
            }
        } else {
            for (int i = 0; i < mre_s->mre->expert_num; i++) {
                rnn_s = mre_s->expert_rnn_s[i];
                rnn_forward_map(rnn_s->rnn_p, mre_s->out_state[n-delay_length],
                        rnn_s->c_inter_state[n-1], rnn_s->c_state[n-1],
                        rnn_s->c_inputsum[n], rnn_s->c_inter_state[n],
                        rnn_s->c_state[n], rnn_s->o_inter_state[n],
                        rnn_s->out_state[n]);
            }
        }
        for (int i = 0; i < mre_s->mre->out_state_size; i++) {
            mre_s->out_state[n][i] = 0;
            for (int j = 0; j < mre_s->mre->expert_num; j++) {
                mre_s->out_state[n][i] += mre_s->gate[j][n] *
                    mre_s->expert_rnn_s[j]->out_state[n][i];
            }
        }
    }
}


void mre_forward_dynamics_in_closed_loop_forall (
        struct mixture_of_rnn_experts *mre,
        int delay_length)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        mre_forward_dynamics_in_closed_loop(mre->mre_s + i, delay_length);
    }
}

static double gauss_func (
        const double *x,
        const double *y,
        int dimension,
        double variance)
{
    double sum = 0;
    for (int i = 0; i < dimension; i++) {
        double d = x[i] - y[i];
        sum += d * d;
    }
    return exp((-sum) / (2*variance)) / pow((2*M_PI)*variance, dimension/2.0);
}


static inline void gmap (
        const struct rnn_state *rnn_s,
        double *generation_likelihood)
{
    const int length = rnn_s->length;
    for (int n = 0; n < length; n++) {
        generation_likelihood[n] = gauss_func(rnn_s->out_state[n],
                rnn_s->teach_state[n], rnn_s->rnn_p->out_state_size,
                rnn_s->rnn_p->variance);
    }
}


void mre_set_generation_likelihood (struct mre_state *mre_s)
{
    const int expert_num = mre_s->mre->expert_num;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < expert_num; i++) {
        gmap(mre_s->expert_rnn_s[i], mre_s->generation_likelihood[i]);
    }
}


void mre_set_joint_likelihood (struct mre_state *mre_s)
{
    const int expert_num = mre_s->mre->expert_num;
    const int length = mre_s->length;
    for (int n = 0; n < length; n++) {
        mre_s->joint_likelihood[n] = 0;
        for (int i = 0; i < expert_num; i++) {
            mre_s->joint_likelihood[n] += mre_s->gate[i][n] *
                mre_s->generation_likelihood[i][n];
        }
    }
}


static inline void dmap (
        const int length,
        const double * const restrict gate,
        const double * const restrict generation_likelihood,
        const double * const restrict joint_likelihood,
        double * restrict discrimination_likelihood)
{
    for (int n = 0; n < length; n++) {
        discrimination_likelihood[n] = (gate[n] * generation_likelihood[n]) /
            joint_likelihood[n];
    }
}

void mre_set_discrimination_likelihood (struct mre_state *mre_s)
{
    const int expert_num = mre_s->mre->expert_num;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < expert_num; i++) {
        dmap(mre_s->length, mre_s->gate[i], mre_s->generation_likelihood[i],
                mre_s->joint_likelihood, mre_s->discrimination_likelihood[i]);
    }
}


static inline void pmap (
        const enum gate_distribution_t gate_prior_distribution,
        const int length,
        const double gamma,
        const double * const restrict beta,
        double * restrict prior_likelihood)
{
    switch (gate_prior_distribution) {
        case NO_DISTRIBUTION:
            for (int n = 0; n < length; n++) {
                prior_likelihood[n] = 1;
            }
            break;
        case GAUSS_DISTRIBUTION:
            {
                double per_gamma2 = 1.0 / (gamma * gamma);
                double c = 1.0 / (sqrt(2*M_PI) * gamma);
                prior_likelihood[0] = 1;
                for (int n = 1; n < length; n++) {
                    double x = beta[n] - beta[n-1];
                    x = x * x;
                    prior_likelihood[n] = exp(-0.5 * x * per_gamma2) * c;
                }
            }
            break;
        case CAUCHY_DISTRIBUTION:
            {
                double gamma2 = gamma * gamma;
                prior_likelihood[0] = 1;
                for (int n = 1; n < length; n++) {
                    double x = beta[n] - beta[n-1];
                    x = x * x;
                    prior_likelihood[n] = gamma / (M_PI * (x + gamma2));
                }
            }
            break;
    }
}

void mre_set_prior_likelihood (struct mre_state *mre_s)
{
    const struct mixture_of_rnn_experts *mre = mre_s->mre;
    const int expert_num = mre->expert_num;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < expert_num; i++) {
        pmap(mre->gate_prior_distribution, mre_s->length, mre->gamma[i],
                mre_s->beta[i], mre_s->prior_likelihood[i]);
    }
}


void mre_set_likelihood_of_expert (
        struct rnn_state *rnn_s,
        double *discrimination_likelihood)
{
    const int length = rnn_s->length;
    const int out_state_size = rnn_s->rnn_p->out_state_size;
    rnn_set_likelihood(rnn_s);
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < out_state_size; i++) {
            rnn_s->delta_likelihood[n][i] *= discrimination_likelihood[n];
            rnn_s->likelihood[n][i] *= discrimination_likelihood[n];
        }
    }
}


void mre_set_likelihood (struct mre_state *mre_s)
{

    mre_set_generation_likelihood(mre_s);
    mre_set_joint_likelihood(mre_s);
    mre_set_discrimination_likelihood(mre_s);
    mre_set_prior_likelihood(mre_s);

    const int expert_num = mre_s->mre->expert_num;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < expert_num; i++) {
        mre_set_likelihood_of_expert(mre_s->expert_rnn_s[i],
                mre_s->discrimination_likelihood[i]);
    }
}


void mre_set_likelihood_forall (struct mixture_of_rnn_experts *mre)
{
#ifdef _OPENMP
    const int total_num = mre->series_num * mre->expert_num;
#pragma omp parallel
    {
#pragma omp for
        for (int n = 0; n < total_num; n++) {
            int i = n / mre->expert_num;
            int j = n % mre->expert_num;
            gmap(mre->mre_s[i].expert_rnn_s[j],
                    mre->mre_s[i].generation_likelihood[j]);
        }
#pragma omp for
        for (int i = 0; i < mre->series_num; i++) {
            mre_set_joint_likelihood(mre->mre_s + i);
        }
#pragma omp for
        for (int n = 0; n < total_num; n++) {
            int i = n / mre->expert_num;
            int j = n % mre->expert_num;
            dmap(mre->mre_s[i].length, mre->mre_s[i].gate[j],
                    mre->mre_s[i].generation_likelihood[j],
                    mre->mre_s[i].joint_likelihood,
                    mre->mre_s[i].discrimination_likelihood[j]);
            pmap(mre->gate_prior_distribution, mre->mre_s[i].length,
                    mre->gamma[j], mre->mre_s[i].beta[j],
                    mre->mre_s[i].prior_likelihood[j]);
            mre_set_likelihood_of_expert(mre->mre_s[i].expert_rnn_s[j],
                    mre->mre_s[i].discrimination_likelihood[j]);
        }
    }
#else
    for (int i = 0; i < mre->series_num; i++) {
        mre_set_likelihood(mre->mre_s + i);
    }
#endif
}


void mre_backward_dynamics (struct mre_state *mre_s)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre_s->mre->expert_num; i++) {
        rnn_backward_dynamics(mre_s->expert_rnn_s[i]);
    }
}


void mre_backward_dynamics_forall (struct mixture_of_rnn_experts *mre)
{
#ifdef _OPENMP
    const int total_num = mre->series_num * mre->expert_num;
#pragma omp parallel for
    for (int n = 0; n < total_num; n++) {
        int i = n / mre->expert_num;
        int j = n % mre->expert_num;
        rnn_backward_dynamics(mre->mre_s[i].expert_rnn_s[j]);
    }
#else
    for (int i = 0; i < mre->series_num; i++) {
        mre_backward_dynamics(mre->mre_s + i);
    }
#endif
}




void mre_forward_backward_dynamics (struct mre_state *mre_s)
{
    mre_forward_dynamics(mre_s);
    mre_set_likelihood(mre_s);
    mre_backward_dynamics(mre_s);
}



void mre_forward_backward_dynamics_forall (struct mixture_of_rnn_experts *mre)
{
    mre_forward_dynamics_forall(mre);
    mre_set_likelihood_forall(mre);
    mre_backward_dynamics_forall(mre);
}


void mre_update_delta_beta (
        struct mre_state *mre_s,
        double momentum)
{
    double sum, per_gamma2 = 0, gamma2 = 0;
    const struct mixture_of_rnn_experts *mre = mre_s->mre;
    const int expert_num = mre->expert_num;
    const int length = mre_s->length;
    const enum gate_distribution_t gate_prior_distribution =
        mre->gate_prior_distribution;

    for (int i = 0; i < expert_num; i++) {
        switch (gate_prior_distribution) {
            case NO_DISTRIBUTION:
                break;
            case GAUSS_DISTRIBUTION:
                per_gamma2 = 1.0 / (mre->gamma[i] * mre->gamma[i]);
                break;
            case CAUCHY_DISTRIBUTION:
                gamma2 = mre->gamma[i] * mre->gamma[i];
                break;
        }
        for (int n = 0; n < length; n++) {
            double delta = (mre_s->gate[i][n] / mre_s->joint_likelihood[n]) *
                (mre_s->generation_likelihood[i][n] -
                 mre_s->joint_likelihood[n]);
            switch (gate_prior_distribution) {
                case NO_DISTRIBUTION:
                    break;
                case GAUSS_DISTRIBUTION:
                    sum = 0;
                    if (n < length - 1) {
                        sum += (mre_s->beta[i][n+1] - mre_s->beta[i][n]);
                    }
                    if (n > 0) {
                        sum += -(mre_s->beta[i][n] - mre_s->beta[i][n-1]);
                    }
                    delta += sum * per_gamma2;
                    break;
                case CAUCHY_DISTRIBUTION:
                    sum = 0;
                    if (n < length - 1) {
                        double d = mre_s->beta[i][n+1] - mre_s->beta[i][n];
                        sum += (2 * d) / (d * d + gamma2);
                    }
                    if (n > 0) {
                        double d = mre_s->beta[i][n] - mre_s->beta[i][n-1];
                        sum += -(2 * d) / (d * d + gamma2);
                    }
                    delta += sum;
                    break;
            }
            mre_s->delta_beta[i][n] = delta + momentum *
                mre_s->delta_beta[i][n];
        }
    }
}


void mre_update_beta_and_gate (
        struct mre_state *mre_s,
        double rho)
{
    for (int n = 0; n < mre_s->length; n++) {
        for (int i = 0; i < mre_s->mre->expert_num; i++) {
            mre_s->beta[i][n] += (rho * mre_s->delta_beta[i][n]);
            assert(isfinite(mre_s->beta[i][n]));
        }
    }

    for (int n = 0; n < mre_s->length; n++) {
        double sum = 0;
        for (int i = 0; i < mre_s->mre->expert_num; i++) {
            sum += exp(mre_s->beta[i][n]);
        }
        for (int i = 0; i < mre_s->mre->expert_num; i++) {
            mre_s->gate[i][n] = exp(mre_s->beta[i][n]) / sum;
        }
    }
}


void mre_update_delta_parameters (
        struct mixture_of_rnn_experts *mre,
        double momentum)
{
    for (int i = 0; i < mre->expert_num; i++) {
        rnn_update_delta_parameters(mre->expert_rnn + i, momentum);
    }
    if (!mre->fixed_gate) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < mre->series_num; i++) {
            mre_update_delta_beta(mre->mre_s + i, momentum);
        }
    }
}


void mre_update_parameters (
        struct mixture_of_rnn_experts *mre,
        double rho_gate,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma)
{
    for (int i = 0; i < mre->expert_num; i++) {
        rnn_update_parameters(mre->expert_rnn + i, rho_weight, rho_tau,
                rho_init, rho_sigma);
    }
    if (!mre->fixed_gate) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < mre->series_num; i++) {
            mre_update_beta_and_gate(mre->mre_s + i, rho_gate);
        }
    }
}


/*
 * This function computes learning of a mixture of rnn experts
 *
 *   @parameter  mre        : mixture of rnn experts
 *   @parameter  rho_gate   : learning rate for gate opening values
 *   @parameter  rho_weight : learning rate for weights and thresholds
 *   @parameter  rho_tau    : learning rate for tau
 *   @parameter  rho_init   : learning rate for initial states
 *   @parameter  rho_sigma  : learning rate for sigma
 *   @parameter  momentum   : momentum of learning
 */
void mre_learn (
        struct mixture_of_rnn_experts *mre,
        double rho_gate,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma,
        double momentum)
{
    mre_forward_backward_dynamics_forall(mre);

    mre_update_delta_parameters(mre, momentum);

    mre_update_parameters(mre, rho_gate, rho_weight, rho_tau, rho_init,
            rho_sigma);
}


/*
 * This function computes learning of a mixture of rnn experts
 * (support automatic scaling of learning rate)
 *
 *   @parameter  mre        : mixture of rnn experts
 *   @parameter  rho        : learning rate
 *   @parameter  momentum   : momentum of learning
 */
void mre_learn_s (
        struct mixture_of_rnn_experts *mre,
        double rho,
        double momentum)
{
    double r = 1.0 / (mre_get_total_length(mre) * mre->out_state_size);
    double rho_weight = r * rho;
    double rho_tau = r * rho;
    double rho_sigma = r * rho;
    double rho_init = rho / mre->out_state_size;
    mre_learn(mre, rho, rho_weight, rho_tau, rho_init, rho_sigma, momentum);
}


#ifdef ENABLE_ADAPTIVE_LEARNING_RATE

void mre_backup_learning_parameters (struct mixture_of_rnn_experts *mre)
{
    for (int i = 0; i < mre->expert_num; i++) {
        rnn_backup_learning_parameters(mre->expert_rnn + i);
    }
    for (int i = 0; i < mre->series_num; i++) {
        struct mre_state *mre_s = mre->mre_s + i;
        memmove(mre_s->tmp_gate, mre_s->gate[0], sizeof(double) *
                mre_s->length * mre->expert_num);
        memmove(mre_s->tmp_beta, mre_s->beta[0], sizeof(double) *
                mre_s->length * mre->expert_num);
    }
}


void mre_restore_learning_parameters (struct mixture_of_rnn_experts *mre)
{
    for (int i = 0; i < mre->expert_num; i++) {
        rnn_restore_learning_parameters(mre->expert_rnn + i);
    }
    for (int i = 0; i < mre->series_num; i++) {
        struct mre_state *mre_s = mre->mre_s + i;
        memmove(mre_s->gate[0], mre_s->tmp_gate, sizeof(double) *
                mre_s->length * mre->expert_num);
        memmove(mre_s->beta[0], mre_s->tmp_beta, sizeof(double) *
                mre_s->length * mre->expert_num);
    }
}


double mre_update_parameters_with_adapt_lr (
        struct mixture_of_rnn_experts *mre,
        double adapt_lr,
        double rho_gate,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma)
{
    double current_error = mre_get_total_error(mre);
    mre_backup_learning_parameters(mre);

    for (int count = 0; count < MAX_ITERATION_IN_ADAPTIVE_LR; count++) {
        mre_update_parameters(mre, rho_gate * adapt_lr, rho_weight * adapt_lr,
                rho_tau * adapt_lr, rho_init * adapt_lr, rho_sigma * adapt_lr);
        mre_forward_dynamics_forall(mre);
        double next_error = mre_get_total_error(mre);
        double rate = next_error / current_error;
        if (rate > MAX_PERF_INC || isnan(rate)) {
            mre_restore_learning_parameters(mre);
            adapt_lr *= LR_DEC;
        } else {
            if (rate < 1) {
                adapt_lr *= LR_INC;
            }
            break;
        }
    }
    return adapt_lr;
}

/*
 * This function computes learning of a mixture of rnn experts
 * (support adaptive learning rate)
 *
 *   @parameter  mre        : mixture of rnn experts
 *   @parameter  adapt_lr   : adaptive learning rate
 *   @parameter  rho_gate   : learning rate for gate opening values
 *   @parameter  rho_weight : learning rate for weights and thresholds
 *   @parameter  rho_tau    : learning rate for tau
 *   @parameter  rho_init   : learning rate for initial states
 *   @parameter  rho_sigma  : learning rate for sigma
 *   @parameter  momentum   : momentum of learning
 *
 *   @return                : adaptive learning rate
 */
double mre_learn_with_adapt_lr (
        struct mixture_of_rnn_experts *mre,
        double adapt_lr,
        double rho_gate,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma,
        double momentum)
{
    mre_forward_backward_dynamics_forall(mre);

    mre_update_delta_parameters(mre, momentum);

    return mre_update_parameters_with_adapt_lr(mre, adapt_lr, rho_gate,
            rho_weight, rho_tau, rho_init, rho_sigma);
}


/*
 * This function computes learning of a mixture of rnn experts
 * (support adaptive learning rate and automatic scaling of learning rate)
 *
 *   @parameter  mre        : mixture of rnn experts
 *   @parameter  adapt_lr   : adaptive learning rate
 *   @parameter  rho        : learning rate
 *   @parameter  momentum   : momentum of learning
 */
double mre_learn_s_with_adapt_lr (
        struct mixture_of_rnn_experts *mre,
        double adapt_lr,
        double rho,
        double momentum)
{
    double r = 1.0 / (mre_get_total_length(mre) * mre->out_state_size);
    double rho_weight = r * rho;
    double rho_tau = r * rho;
    double rho_sigma = r * rho;
    double rho_init = rho / mre->out_state_size;
    return mre_learn_with_adapt_lr(mre, adapt_lr, rho, rho_weight, rho_tau,
            rho_init, rho_sigma, momentum);
}


#endif // ENABLE_ADAPTIVE_LEARNING_RATE




void mre_update_prior_strength (
        struct mixture_of_rnn_experts *mre,
        double lambda,
        double alpha)
{
    double likelihood;
    struct rnn_parameters *rnn_p;

    mre_forward_dynamics_forall(mre);
    mre_set_likelihood_forall(mre);

    for (int i = 0; i < mre->expert_num; i++) {
        likelihood = 0;
        for (int j = 0; j < mre->series_num; j++) {
            for (int n = 0; n < mre->mre_s[j].length; n++) {
                likelihood += mre->mre_s[j].discrimination_likelihood[i][n] *
                    mre->mre_s[j].joint_likelihood[n];
            }
        }
        rnn_p = &mre->expert_rnn[i].rnn_p;
        rnn_p->prior_strength = lambda * rnn_p->prior_strength +
            alpha * likelihood;
        rnn_reset_prior_distribution(rnn_p);
    }
}

