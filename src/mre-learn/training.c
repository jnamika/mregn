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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "training.h"
#include "mre.h"
#include "print.h"


#ifndef SEED_TRANSIENT
#define SEED_TRANSIENT 100000
#endif

static void init_mre (
        const struct general_parameters *gp,
        const struct target_reader *t_reader,
        struct mixture_of_rnn_experts *mre);

static void save_mre (
        const struct general_parameters *gp,
        const struct mixture_of_rnn_experts *mre);

static void load_mre (
        struct general_parameters *gp,
        const struct target_reader *t_reader,
        struct mixture_of_rnn_experts *mre);

static void free_mre (struct mixture_of_rnn_experts *mre);

static void set_parameters_to_recurrent_neural_network (
        const struct general_parameters *gp,
        struct recurrent_neural_network *rnn);

static void set_parameters_to_mixture_of_rnn_experts (
        const struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre);

/******************************************************************************/
/************** Training Main *************************************************/
/******************************************************************************/

static void init_training_main (
        struct general_parameters *gp,
        const struct target_reader *t_reader,
        struct mixture_of_rnn_experts *mre,
        struct output_files *fp_list)
{
    init_genrand(gp->mp.seed);
    for (int i = 0; i < SEED_TRANSIENT; i++) {
        xor128();
    }

    int has_load_file = strlen(gp->iop.load_filename);
    if (has_load_file) {
        load_mre(gp, t_reader, mre);
    } else {
        init_mre(gp, t_reader, mre);
    }

    set_parameters_to_mixture_of_rnn_experts(gp, mre);

    if (!has_load_file || t_reader->num > 0) {
        init_output_files(gp, mre, fp_list, "w");
    } else {
        init_output_files(gp, mre, fp_list, "a");
    }
}


static void fini_training_main (
        const struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre,
        struct output_files *fp_list)
{
    if (strlen(gp->iop.save_filename) > 0) {
        save_mre(gp, mre);
    }
    free_mre(mre);
    free_output_files(fp_list);
}


void training_main (
        struct general_parameters *gp,
        const struct target_reader *t_reader)
{
    struct mixture_of_rnn_experts mre;
    struct output_files fp_list;

    init_training_main(gp, t_reader, &mre, &fp_list);

    if (strlen(gp->iop.load_filename) == 0 || t_reader->num > 0) {
        print_training_main_begin(gp, &mre, &fp_list);
    }

    for (long epoch = gp->inp.init_epoch; epoch <= gp->mp.epoch_size; epoch++) {
        if (!gp->mp.use_adaptive_lr) {
            mre_learn_s(&mre, gp->mp.rho, gp->mp.momentum);
        } else {
#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
            gp->inp.adapt_lr = mre_learn_s_with_adapt_lr(&mre, gp->inp.adapt_lr,
                    gp->mp.rho, gp->mp.momentum);
#else
            print_error_msg("option `use_adaptive_lr' is not supported");
            exit(EXIT_FAILURE);
#endif
        }
        if (gp->iop.verbose) {
            printf("epoch = %ld\n", epoch);
            fflush(stdout);
        }
        print_training_main_loop(epoch, gp, &mre, &fp_list);
    }

    fini_training_main(gp, &mre, &fp_list);
}




/******************************************************************************/
/****************** Initialization and Free ***********************************/
/******************************************************************************/

static void set_parameters_to_recurrent_neural_network (
        const struct general_parameters *gp,
        struct recurrent_neural_network *rnn)
{
    struct rnn_parameters *rnn_p = &rnn->rnn_p;
    int total_length = rnn_get_total_length(rnn);

    rnn_p->fixed_weight = gp->mp.fixed_weight;
    rnn_p->fixed_threshold = gp->mp.fixed_threshold;
    rnn_p->fixed_tau = gp->mp.fixed_tau;
    rnn_p->fixed_init_c_state = gp->mp.fixed_init_c_state;
    rnn_p->fixed_sigma = gp->mp.fixed_sigma;

    if (strlen(gp->iop.load_filename) == 0) {
        rnn_p->output_type = STANDARD_TYPE;
        for (int i = 0; i < rnn_p->c_state_size; i++) {
            for (int j = 0; j < rnn_p->in_state_size; j++) {
                if (gp->inp.connectivity_ci[i][j] <= genrand_real2()) {
                    gp->inp.has_connection_ci[i][j] = 0;
                }
            }
            rnn_set_connection(rnn_p->in_state_size, rnn_p->connection_ci[i],
                    gp->inp.has_connection_ci[i]);
            for (int j = 0; j < rnn_p->c_state_size; j++) {
                if (gp->inp.connectivity_cc[i][j] <= genrand_real2()) {
                    gp->inp.has_connection_cc[i][j] = 0;
                }
            }
            rnn_set_connection(rnn_p->c_state_size, rnn_p->connection_cc[i],
                    gp->inp.has_connection_cc[i]);
        }
        for (int i = 0; i < rnn_p->out_state_size; i++) {
            for (int j = 0; j < rnn_p->c_state_size; j++) {
                if (gp->inp.connectivity_oc[i][j] <= genrand_real2()) {
                    gp->inp.has_connection_oc[i][j] = 0;
                }
            }
            rnn_set_connection(rnn_p->c_state_size, rnn_p->connection_oc[i],
                    gp->inp.has_connection_oc[i]);
        }
        rnn_reset_weight_by_connection(rnn_p);
        memcpy(rnn_p->const_init_c, gp->inp.const_init_c, rnn_p->c_state_size *
                sizeof(int));
        rnn_set_tau(rnn_p, gp->inp.init_tau);
        rnn_set_sigma(rnn_p, gp->mp.init_sigma);
        rnn_p->prior_strength = gp->mp.prior_strength * rnn_p->out_state_size *
            total_length;
        rnn_reset_prior_distribution(rnn_p);
    }
}

static void set_parameters_to_mixture_of_rnn_experts (
        const struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre)
{
    for (int i = 0; i < mre->expert_num; i++) {
        set_parameters_to_recurrent_neural_network(gp, mre->expert_rnn + i);
    }
    mre->fixed_gate = gp->mp.fixed_gate;
    if (strlen(gp->iop.load_filename) == 0) {
        for (int i = 0; i < mre->expert_num; i++) {
            mre->gamma[i] = gp->mp.gamma;
        }
        mre->gate_prior_distribution =
            (enum gate_distribution_t)gp->mp.gate_prior_distribution;
    }
}



static void init_mre (
        const struct general_parameters *gp,
        const struct target_reader *t_reader,
        struct mixture_of_rnn_experts *mre)
{
    init_mixture_of_rnn_experts(mre, gp->mp.expert_num, t_reader->dimension,
            gp->mp.c_state_size, t_reader->dimension);
    for (int i = 0; i < t_reader->num; i++) {
        if (t_reader->t_list[i].length <= gp->mp.delay_length) {
            print_error_msg("length of target time series must be greater "
                    "than time delay.");
            exit(EXIT_FAILURE);
        }
        mre_add_target(mre, t_reader->t_list[i].length - gp->mp.delay_length,
                (const double* const*)t_reader->t_list[i].target,
                (const double* const*)t_reader->t_list[i].target +
                gp->mp.delay_length);
    }
}


static void save_mre (
        const struct general_parameters *gp,
        const struct mixture_of_rnn_experts *mre)
{
    FILE *fp;
    long init_epoch;
    if (gp->mp.epoch_size >= 0) {
        init_epoch = gp->mp.epoch_size + 1;
    } else {
        init_epoch = 0;
    }
    if ((fp = fopen(gp->iop.save_filename, "wb")) == NULL) {
        print_error_msg("cannot open %s", gp->iop.save_filename);
        exit(EXIT_FAILURE);
    }
    FWRITE(&gp->mp.delay_length, 1, fp);
    fwrite_mixture_of_rnn_experts(mre, fp);
    FWRITE(&gp->inp.adapt_lr, 1, fp);
    FWRITE(&init_epoch, 1, fp);
    fclose(fp);
}


static void reset_target_of_mre (
        struct general_parameters *gp,
        const struct target_reader *t_reader,
        struct mixture_of_rnn_experts *mre)
{
    mre_update_prior_strength(mre, gp->mp.lambda, gp->mp.alpha);
    for (int i = 0; i < mre->expert_num; i++) {
        rnn_reset_delta_parameters(&mre->expert_rnn[i].rnn_p);
    }
    mre_clean_target(mre);
    if (t_reader->dimension != mre->out_state_size) {
        print_error_msg("t_reader->dimension(=%d) != "
                "rnn->rnn_p.out_state_size(=%d)", t_reader->dimension,
                mre->out_state_size);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < t_reader->num; i++) {
        if (t_reader->t_list[i].length <= gp->mp.delay_length) {
            print_error_msg("length of target time series must be greater "
                    "than time delay.");
            exit(EXIT_FAILURE);
        }
        mre_add_target(mre, t_reader->t_list[i].length,
                (const double* const*)t_reader->t_list[i].target,
                (const double* const*)t_reader->t_list[i].target +
                gp->mp.delay_length);
    }
    gp->inp.init_epoch = 0;
}


static void load_mre (
        struct general_parameters *gp,
        const struct target_reader *t_reader,
        struct mixture_of_rnn_experts *mre)
{
    FILE *fp;
    if ((fp= fopen(gp->iop.load_filename, "rb")) == NULL) {
        print_error_msg("cannot open %s", gp->iop.load_filename);
        exit(EXIT_FAILURE);
    }
    FREAD(&gp->mp.delay_length, 1, fp);
    fread_mixture_of_rnn_experts(mre, fp);
    FREAD(&gp->inp.adapt_lr, 1, fp);
    FREAD(&gp->inp.init_epoch, 1, fp);
    fclose(fp);
    if (t_reader->num > 0) {
        reset_target_of_mre(gp, t_reader, mre);
    }
}


static void free_mre (struct mixture_of_rnn_experts *mre)
{
    free_mixture_of_rnn_experts(mre);
}


