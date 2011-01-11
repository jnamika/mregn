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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "training.h"
#include "rnn.h"
#include "print.h"


#ifndef SEED_TRANSIENT
#define SEED_TRANSIENT 100000
#endif

static void init_rnn (
        const struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *rnn);

static void save_rnn (
        const struct general_parameters *gp,
        const struct recurrent_neural_network *rnn);

static void load_rnn (
        struct general_parameters *gp,
        struct recurrent_neural_network *rnn);

static void free_rnn (struct recurrent_neural_network *rnn);

static void set_parameters_to_recurrent_neural_network (
        const struct general_parameters *gp,
        struct recurrent_neural_network *rnn);

/******************************************************************************/
/************** Training Main *************************************************/
/******************************************************************************/

static void init_training_main (
        struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *rnn,
        struct output_files *fp_list)
{
    init_genrand(gp->mp.seed);
    for (int i = 0; i < SEED_TRANSIENT; i++) {
        genrand_int32();
    }

    if (gp->mp.gn_delay_length <= 0) {
        gp->mp.gn_delay_length = gp->mp.mre_delay_length;
    }

    int has_load_file = strlen(gp->iop.load_filename);
    if (has_load_file) {
        load_rnn(gp, rnn);
    } else {
        init_rnn(gp, mre, rnn);
    }

    set_parameters_to_recurrent_neural_network(gp, rnn);

    if (!has_load_file) {
        init_output_files(gp, rnn, fp_list, "w");
    } else {
        init_output_files(gp, rnn, fp_list, "a");
    }
}


static void fini_training_main (
        const struct general_parameters *gp,
        struct recurrent_neural_network *rnn,
        struct output_files *fp_list)
{
    if (strlen(gp->iop.save_filename) > 0) {
        save_rnn(gp, rnn);
    }
    free_rnn(rnn);
    free_output_files(fp_list);
}


void training_main (
        struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre)
{
    struct recurrent_neural_network rnn;
    struct output_files fp_list;

    init_training_main(gp, mre, &rnn, &fp_list);

    if (strlen(gp->iop.load_filename) == 0) {
        print_training_main_begin(gp, &rnn, &fp_list);
    }

    for (long epoch = gp->inp.init_epoch; epoch <= gp->mp.epoch_size; epoch++) {
        if (!gp->mp.use_adaptive_lr) {
            rnn_learn_s(&rnn, gp->mp.rho, gp->mp.momentum);
        } else {
#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
            gp->inp.adapt_lr = rnn_learn_s_with_adapt_lr(&rnn, gp->inp.adapt_lr,
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
        print_training_main_loop(epoch, gp, mre, &rnn, &fp_list);
    }

    fini_training_main(gp, &rnn, &fp_list);
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

    if (strlen(gp->iop.load_filename) == 0) {
        rnn_p->output_type = SOFTMAX_TYPE;
        for (int i = 0; i < rnn_p->c_state_size; i++) {
            rnn_set_connection(rnn_p->in_state_size, rnn_p->connection_ci[i],
                    gp->inp.has_connection_ci[i]);
            rnn_set_connection(rnn_p->c_state_size, rnn_p->connection_cc[i],
                    gp->inp.has_connection_cc[i]);
        }
        for (int i = 0; i < rnn_p->out_state_size; i++) {
            rnn_set_connection(rnn_p->c_state_size, rnn_p->connection_oc[i],
                    gp->inp.has_connection_oc[i]);
        }
        rnn_reset_weight_by_connection(rnn_p);
        memcpy(rnn_p->const_init_c, gp->inp.const_init_c, rnn_p->c_state_size *
                sizeof(int));
        rnn_set_tau(rnn_p, gp->inp.init_tau);
        rnn_p->prior_strength = gp->mp.prior_strength * rnn_p->out_state_size *
            total_length;
        rnn_reset_prior_distribution(rnn_p);
    }
}

static void get_target_sequence (
        struct mre_state *mre_s,
        double ***sequence,
        int *size)
{
    if (*size < mre_s->length) {
        REALLOC(*sequence, mre_s->length);
        int dim = mre_s->mre->expert_num + mre_s->mre->in_state_size;
        if (*size > 0) {
            REALLOC((*sequence)[0], mre_s->length * dim);
        } else {
            MALLOC((*sequence)[0], mre_s->length * dim);
        }
        *size = mre_s->length;
        for (int n = 0; n < *size; n++) {
            (*sequence)[n] = (*sequence)[0] + (n * dim);
        }
    }
    for (int n = 0; n < mre_s->length; n++) {
        for (int i = 0; i < mre_s->mre->expert_num; i++) {
            (*sequence)[n][i] = mre_s->gate[i][n];
        }
        for (int i = 0; i < mre_s->mre->in_state_size; i++) {
            (*sequence)[n][i+mre_s->mre->expert_num] =
                mre_s->expert_rnn_s[0]->in_state[n][i];
        }
    }
}

static void init_rnn (
        const struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *rnn)
{
    const int in_state_size = mre->expert_num + mre->in_state_size;
    init_recurrent_neural_network(rnn, in_state_size, gp->mp.c_state_size,
            mre->expert_num);
    int size = 0;
    double **sequence = NULL;
    for (int i = 0; i < mre->series_num; i++) {
        if (mre->mre_s[i].length <= gp->mp.gn_delay_length) {
            print_error_msg("length of target data must be greater than "
                    "time delay in a self-feedback");
            exit(EXIT_FAILURE);
        }
        get_target_sequence(mre->mre_s + i, &sequence, &size);
        rnn_add_target(rnn, mre->mre_s[i].length - gp->mp.gn_delay_length,
                sequence, sequence + gp->mp.gn_delay_length);
    }
    free(sequence[0]);
    free(sequence);
    for (int i = 0; i < mre->series_num; i++) {
        mre->mre_s[i].length = rnn->rnn_s[i].length;
        for (int j = 0; j < mre->expert_num; j++) {
            mre->expert_rnn[j].rnn_s[i].length = rnn->rnn_s[i].length;
        }
    }
}


static void save_rnn (
        const struct general_parameters *gp,
        const struct recurrent_neural_network *rnn)
{
    FILE *fp;
    long init_epoch;
    init_epoch = gp->mp.epoch_size + 1;
    if ((fp = fopen(gp->iop.save_filename, "wb")) == NULL) {
        print_error_msg("cannot open %s", gp->iop.save_filename);
        exit(EXIT_FAILURE);
    }
    FWRITE(&gp->mp.gn_delay_length, 1, fp);
    fwrite_recurrent_neural_network(rnn, fp);
    FWRITE(&gp->inp.adapt_lr, 1, fp);
    FWRITE(&init_epoch, 1, fp);
    fclose(fp);
}


static void load_rnn (
        struct general_parameters *gp,
        struct recurrent_neural_network *rnn)
{
    FILE *fp;
    if ((fp= fopen(gp->iop.load_filename, "rb")) == NULL) {
        print_error_msg("cannot open %s", gp->iop.load_filename);
        exit(EXIT_FAILURE);
    }
    FREAD(&gp->mp.gn_delay_length, 1, fp);
    fread_recurrent_neural_network(rnn, fp);
    FREAD(&gp->inp.adapt_lr, 1, fp);
    FREAD(&gp->inp.init_epoch, 1, fp);
    fclose(fp);
}


static void free_rnn (struct recurrent_neural_network *rnn)
{
    free_recurrent_neural_network(rnn);
}


