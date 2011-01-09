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

#ifndef MRE_H
#define MRE_H


#include "rnn.h"

typedef enum gate_distribution_t {
    NO_DISTRIBUTION,
    GAUSS_DISTRIBUTION,
    CAUCHY_DISTRIBUTION
} gate_distribution_t;


typedef struct mre_state {
    struct mixture_of_rnn_experts *mre;
    int length;

    double **gate;
    double **beta;
    double **delta_beta;

    double *joint_likelihood;
    double **generation_likelihood;
    double **discrimination_likelihood;
    double **prior_likelihood;

    double **out_state;

    struct rnn_state **expert_rnn_s;

#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    double *tmp_gate;
    double *tmp_beta;
#endif
} mre_state;


typedef struct mixture_of_rnn_experts {
    int expert_num;
    int series_num;

    int in_state_size;
    int out_state_size;

    /* If fixed_gate != 0, then gate does not change by learning. */
    int fixed_gate;
    enum gate_distribution_t gate_prior_distribution;
    double *gamma;

    struct recurrent_neural_network *expert_rnn;
    struct mre_state *mre_s;
} mixture_of_rnn_experts;


void init_mre_state (
        struct mre_state *mre_s,
        struct mixture_of_rnn_experts *mre,
        int length);

void free_mre_state (struct mre_state *mre_s);


void init_mixture_of_rnn_experts (
        struct mixture_of_rnn_experts *mre,
        int expert_num,
        int in_state_size,
        int c_state_size,
        int out_state_size);

void free_mixture_of_rnn_experts (struct mixture_of_rnn_experts *mre);

void mre_add_target (
        struct mixture_of_rnn_experts *mre,
        int length,
        double **input,
        double **target);

void mre_clean_target (struct mixture_of_rnn_experts *mre);

void mre_state_alloc (struct mre_state *mre_s);



void fwrite_mre_state (
        const struct mre_state *mre_s,
        FILE *fp);

void fread_mre_state (
        struct mre_state *mre_s,
        FILE *fp);

void fwrite_mixture_of_rnn_experts (
        const struct mixture_of_rnn_experts *mre,
        FILE *fp);

void fread_mixture_of_rnn_experts (
        struct mixture_of_rnn_experts *mre,
        FILE *fp);

int mre_get_total_length (const struct mixture_of_rnn_experts *mre);

double mre_get_error (const struct mre_state *mre_s);
double mre_get_total_error (const struct mixture_of_rnn_experts *mre);

double mre_get_joint_likelihood (const struct mre_state *mre_s);
double mre_get_total_joint_likelihood (
        const struct mixture_of_rnn_experts *mre);

double mre_get_prior_likelihood (const struct mre_state *mre_s);
double mre_get_total_prior_likelihood (
        const struct mixture_of_rnn_experts *mre);


void mre_forward_dynamics (struct mre_state *mre_s);

void mre_forward_dynamics_forall (struct mixture_of_rnn_experts *mre);

void mre_forward_dynamics_in_closed_loop (
        struct mre_state *mre_s,
        int delay_length);

void mre_forward_dynamics_in_closed_loop_forall (
        struct mixture_of_rnn_experts *mre,
        int delay_length);


void mre_set_generation_likelihood (struct mre_state *mre_s);

void mre_set_joint_likelihood (struct mre_state *mre_s);

void mre_set_discrimination_likelihood (struct mre_state *mre_s);

void mre_set_prior_likelihood (struct mre_state *mre_s);

void mre_set_likelihood_of_expert (
        struct rnn_state *rnn_s,
        double *discrimination_likelihood);

void mre_set_likelihood (struct mre_state *mre_s);

void mre_set_likelihood_forall (struct mixture_of_rnn_experts *mre);

void mre_backward_dynamics (struct mre_state *mre_s);

void mre_backward_dynamics_forall (struct mixture_of_rnn_experts *mre);


void mre_forward_backward_dynamics (struct mre_state *mre_s);

void mre_forward_backward_dynamics_forall (struct mixture_of_rnn_experts *mre);


void mre_update_delta_beta (
        struct mre_state *mre_s,
        double momentum);

void mre_update_beta_and_gate (
        struct mre_state *mre_s,
        double rho);

void mre_update_delta_parameters (
        struct mixture_of_rnn_experts *mre,
        double momentum);

void mre_update_parameters (
        struct mixture_of_rnn_experts *mre,
        double rho_gate,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma);

void mre_learn (
        struct mixture_of_rnn_experts *mre,
        double rho_gate,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma,
        double momentum);

void mre_learn_s (
        struct mixture_of_rnn_experts *mre,
        double rho,
        double momentum);

#ifdef ENABLE_ADAPTIVE_LEARNING_RATE

void mre_backup_learning_parameters (struct mixture_of_rnn_experts *mre);
void mre_restore_learning_parameters (struct mixture_of_rnn_experts *mre);

double mre_update_parameters_with_adapt_lr (
        struct mixture_of_rnn_experts *mre,
        double adapt_lr,
        double rho_gate,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma);

double mre_learn_with_adapt_lr (
        struct mixture_of_rnn_experts *mre,
        double adapt_lr,
        double rho_gate,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma,
        double momentum);

double mre_learn_s_with_adapt_lr (
        struct mixture_of_rnn_experts *mre,
        double adapt_lr,
        double rho,
        double momentum);

#endif

void mre_update_prior_strength (
        struct mixture_of_rnn_experts *mre,
        double lambda,
        double alpha);

#endif

