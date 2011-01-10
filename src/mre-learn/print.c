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
#include "print.h"



static void fopen_array (
        FILE **fp_array,
        int size,
        const char *template_filename,
        const char *mode)
{
    char str[7], *filename, *p;
    int length = strlen(template_filename);
    MALLOC(filename, length + 1);
    strcpy(filename, template_filename);
    p = strstr(filename, "XXXXXX");
    if (p == NULL) {
        REALLOC(filename, length + 8);
        filename[length] = '.';
        filename[length + 7] = '\0';
        p = filename + length + 1;
    }
    for (int i = 0; i < size; i++) {
        snprintf(str, sizeof(str), "%.6d", i);
        memmove(p, str, 6);
        fp_array[i] = fopen(filename, mode);
        if (fp_array[i] == NULL) {
            print_error_msg();
            goto error;
        }
    }
    free(filename);
    return;
error:
    exit(EXIT_FAILURE);
}


void init_output_files (
        const struct general_parameters *gp,
        const struct mixture_of_rnn_experts *mre,
        struct output_files *fp_list,
        const char *mode)
{
    fp_list->array_size = mre->series_num;
    if (strlen(gp->iop.state_filename) > 0) {
        MALLOC(fp_list->fp_wstate_array, fp_list->array_size);
        fopen_array(fp_list->fp_wstate_array, fp_list->array_size,
                gp->iop.state_filename, mode);
    } else {
        fp_list->fp_wstate_array = NULL;
    }

    if (strlen(gp->iop.closed_state_filename) > 0) {
        MALLOC(fp_list->fp_wclosed_state_array, fp_list->array_size);
        fopen_array(fp_list->fp_wclosed_state_array, fp_list->array_size,
                gp->iop.closed_state_filename, mode);
    } else {
        fp_list->fp_wclosed_state_array = NULL;
    }

    if (strlen(gp->iop.gate_filename) > 0) {
        MALLOC(fp_list->fp_wgate_array, fp_list->array_size);
        fopen_array(fp_list->fp_wgate_array, fp_list->array_size,
                gp->iop.gate_filename, mode);
    } else {
        fp_list->fp_wgate_array = NULL;
    }

    if (strlen(gp->iop.weight_filename) > 0) {
        fp_list->fp_wweight = fopen(gp->iop.weight_filename, mode);
        if (fp_list->fp_wweight == NULL) goto error;
    } else {
        fp_list->fp_wweight = NULL;
    }
    if (strlen(gp->iop.threshold_filename) > 0) {
        fp_list->fp_wthreshold = fopen(gp->iop.threshold_filename, mode);
        if (fp_list->fp_wthreshold == NULL) goto error;
    } else {
        fp_list->fp_wthreshold = NULL;
    }
    if (strlen(gp->iop.tau_filename) > 0) {
        fp_list->fp_wtau = fopen(gp->iop.tau_filename, mode);
        if (fp_list->fp_wtau == NULL) goto error;
    } else {
        fp_list->fp_wtau = NULL;
    }
    if (strlen(gp->iop.sigma_filename) > 0) {
        fp_list->fp_wsigma = fopen(gp->iop.sigma_filename, mode);
        if (fp_list->fp_wsigma == NULL) goto error;
    } else {
        fp_list->fp_wsigma = NULL;
    }
    if (strlen(gp->iop.init_filename) > 0) {
        fp_list->fp_winit = fopen(gp->iop.init_filename, mode);
        if (fp_list->fp_winit == NULL) goto error;
    } else {
        fp_list->fp_winit = NULL;
    }
    if (strlen(gp->iop.adapt_lr_filename) > 0 && gp->mp.use_adaptive_lr) {
        fp_list->fp_wadapt_lr = fopen(gp->iop.adapt_lr_filename, mode);
        if (fp_list->fp_wadapt_lr == NULL) goto error;
    } else {
        fp_list->fp_wadapt_lr = NULL;
    }
    if (strlen(gp->iop.error_filename) > 0) {
        fp_list->fp_werror = fopen(gp->iop.error_filename, mode);
        if (fp_list->fp_werror == NULL) goto error;
    } else {
        fp_list->fp_werror = NULL;
    }
    if (strlen(gp->iop.closed_error_filename) > 0) {
        fp_list->fp_wclosed_error = fopen(gp->iop.closed_error_filename, mode);
        if (fp_list->fp_wclosed_error == NULL) goto error;
    } else {
        fp_list->fp_wclosed_error = NULL;
    }
    return;
error:
    print_error_msg();
    exit(EXIT_FAILURE);
}

void free_output_files (struct output_files *fp_list)
{
    if (fp_list->fp_wstate_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fclose(fp_list->fp_wstate_array[i]);
        }
        free(fp_list->fp_wstate_array);
    }
    if (fp_list->fp_wclosed_state_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fclose(fp_list->fp_wclosed_state_array[i]);
        }
        free(fp_list->fp_wclosed_state_array);
    }
    if (fp_list->fp_wgate_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fclose(fp_list->fp_wgate_array[i]);
        }
        free(fp_list->fp_wgate_array);
    }
    if (fp_list->fp_wweight) {
        fclose(fp_list->fp_wweight);
    }
    if (fp_list->fp_wthreshold) {
        fclose(fp_list->fp_wthreshold);
    }
    if (fp_list->fp_wtau) {
        fclose(fp_list->fp_wtau);
    }
    if (fp_list->fp_wsigma) {
        fclose(fp_list->fp_wsigma);
    }
    if (fp_list->fp_winit) {
        fclose(fp_list->fp_winit);
    }
    if (fp_list->fp_wadapt_lr) {
        fclose(fp_list->fp_wadapt_lr);
    }
    if (fp_list->fp_werror) {
        fclose(fp_list->fp_werror);
    }
    if (fp_list->fp_wclosed_error) {
        fclose(fp_list->fp_wclosed_error);
    }
}

static void print_general_parameters (
        FILE *fp,
        const struct general_parameters *gp)
{
    fprintf(fp, "# seed = %lu\n", gp->mp.seed);
    fprintf(fp, "# epoch_size = %ld\n", gp->mp.epoch_size);
    if (gp->mp.use_adaptive_lr) {
        fprintf(fp, "# use_adaptive_lr\n");
    }
    fprintf(fp, "# rho = %f\n", gp->mp.rho);
    fprintf(fp, "# momentum = %f\n", gp->mp.momentum);
    fprintf(fp, "# delay_length = %d\n", gp->mp.delay_length);
    fprintf(fp, "# lambda = %f\n", gp->mp.lambda);
    fprintf(fp, "# alpha = %f\n", gp->mp.alpha);
}



static void print_rnn_parameters (
        FILE *fp,
        const struct recurrent_neural_network *rnn)
{
    fprintf(fp, "# in_state_size = %d\n", rnn->rnn_p.in_state_size);
    fprintf(fp, "# c_state_size = %d\n", rnn->rnn_p.c_state_size);
    fprintf(fp, "# out_state_size = %d\n", rnn->rnn_p.out_state_size);
    if (rnn->rnn_p.output_type == STANDARD_TYPE) {
        fprintf(fp, "# output_type = STANDARD_TYPE\n");
    } else if (rnn->rnn_p.output_type == SOFTMAX_TYPE) {
        fprintf(fp, "# output_type = SOFTMAX_TYPE\n");
        for (int c = 0; c < rnn->rnn_p.softmax_group_num; c++) {
            fprintf(fp, "# group%d = ", c);
            for (int i = 0; i < rnn->rnn_p.out_state_size; i++) {
                if (rnn->rnn_p.softmax_group_id[i] == c) {
                    fprintf(fp, "%d,", i);
                }
            }
            fprintf(fp, "\n");
        }
    }
    if (rnn->rnn_p.fixed_weight) {
        fprintf(fp, "# fixed_weight\n");
    }
    if (rnn->rnn_p.fixed_threshold) {
        fprintf(fp, "# fixed_threshold\n");
    }
    if (rnn->rnn_p.fixed_tau) {
        fprintf(fp, "# fixed_tau\n");
    }
    if (rnn->rnn_p.fixed_init_c_state) {
        fprintf(fp, "# fixed_init_c_state\n");
    }
    if (rnn->rnn_p.fixed_sigma) {
        fprintf(fp, "# fixed_sigma\n");
    }

    fprintf(fp, "# target_num = %d\n", rnn->series_num);
    for (int i = 0; i < rnn->series_num; i++) {
        fprintf(fp, "# target %d\tlength = %d\n", i, rnn->rnn_s[i].length);
    }
    fprintf(fp, "# prior_strength = %f\n", rnn->rnn_p.prior_strength);

    const struct rnn_parameters *rnn_p = &rnn->rnn_p;
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        fprintf(fp, "# const_init_c[%d] = %d\n", i, rnn_p->const_init_c[i]);
    }

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        fprintf(fp, "# connection_weight_ci[%d] = ", i);
        int I = 0;
        while (rnn_p->connection_ci[i][I].begin != -1) {
            int begin = rnn_p->connection_ci[i][I].begin;
            int end = rnn_p->connection_ci[i][I].end;
            fprintf(fp, "(%d,%d)", begin, end);
            I++;
        }
        fprintf(fp, "\n");
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        fprintf(fp, "# connection_weight_cc[%d] = ", i);
        int I = 0;
        while (rnn_p->connection_cc[i][I].begin != -1) {
            int begin = rnn_p->connection_cc[i][I].begin;
            int end = rnn_p->connection_cc[i][I].end;
            fprintf(fp, "(%d,%d)", begin, end);
            I++;
        }
        fprintf(fp, "\n");
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        fprintf(fp, "# connection_weight_oc[%d] = ", i);
        int I = 0;
        while (rnn_p->connection_oc[i][I].begin != -1) {
            int begin = rnn_p->connection_oc[i][I].begin;
            int end = rnn_p->connection_oc[i][I].end;
            fprintf(fp, "(%d,%d)", begin, end);
            I++;
        }
        fprintf(fp, "\n");
    }
}

static void print_mre_parameters (
        FILE *fp,
        const struct mixture_of_rnn_experts *mre)
{
    fprintf(fp, "# expert_num = %d\n", mre->expert_num);
    fprintf(fp, "# target_num = %d\n", mre->series_num);
    if (mre->fixed_gate) {
        fprintf(fp, "# fixed_gate\n");
    }
    fprintf(fp, "# gate_prior_distribution = %d\n",
            (int)mre->gate_prior_distribution);
    for (int i = 0; i < mre->expert_num; i++) {
        fprintf(fp, "# gamma[%d] = %f\n", i, mre->gamma[i]);
    }
    for (int i = 0; i < mre->expert_num; i++) {
        print_rnn_parameters(fp, mre->expert_rnn + i);
    }
}


static void print_mre_weight (
        FILE *fp,
        long epoch,
        const struct mixture_of_rnn_experts *mre)
{
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < mre->expert_num; i++) {
        const struct rnn_parameters *rnn_p = &mre->expert_rnn[i].rnn_p;
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            for (int k = 0; k < rnn_p->in_state_size; k++) {
                fprintf(fp, "\t%f", rnn_p->weight_ci[j][k]);
            }
            for (int k = 0; k < rnn_p->c_state_size; k++) {
                fprintf(fp, "\t%f", rnn_p->weight_cc[j][k]);
            }
        }
        for (int j = 0; j < rnn_p->out_state_size; j++) {
            for (int k = 0; k < rnn_p->c_state_size; k++) {
                fprintf(fp, "\t%f", rnn_p->weight_oc[j][k]);
            }
        }
    }
    fprintf(fp, "\n");
}

static void print_mre_threshold (
        FILE *fp,
        long epoch,
        const struct mixture_of_rnn_experts *mre)
{
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < mre->expert_num; i++) {
        const struct rnn_parameters *rnn_p = &mre->expert_rnn[i].rnn_p;
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            fprintf(fp, "\t%f", rnn_p->threshold_c[j]);
        }
        for (int j = 0; j < rnn_p->out_state_size; j++) {
            fprintf(fp, "\t%f", rnn_p->threshold_o[j]);
        }
    }
    fprintf(fp, "\n");
}


static void print_mre_tau (
        FILE *fp,
        long epoch,
        const struct mixture_of_rnn_experts *mre)
{
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < mre->expert_num; i++) {
        const struct rnn_parameters *rnn_p = &mre->expert_rnn[i].rnn_p;
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            fprintf(fp, "\t%g", rnn_p->tau[j]);
        }
    }
    fprintf(fp, "\n");
}

static void print_mre_sigma (
        FILE *fp,
        long epoch,
        const struct mixture_of_rnn_experts *mre)
{
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < mre->expert_num; i++) {
        const struct rnn_parameters *rnn_p = &mre->expert_rnn[i].rnn_p;
        fprintf(fp, "\t%f\t%f", rnn_p->sigma, rnn_p->variance);
    }
    fprintf(fp, "\n");
}


static void print_mre_init (
        FILE *fp,
        long epoch,
        const struct mixture_of_rnn_experts *mre)
{
    fprintf(fp, "# epoch = %ld\n", epoch);
    for (int i = 0; i < mre->series_num; i++) {
        fprintf(fp, "%d", i);
        const struct rnn_state *rnn_s = mre->mre_s[i].expert_rnn_s[0];
        for (int j = 0; j < mre->expert_num; j++) {
            rnn_s = mre->mre_s[i].expert_rnn_s[j];
            for (int k = 0; k < rnn_s->rnn_p->c_state_size; k++) {
                fprintf(fp, "\t%f", rnn_s->init_c_inter_state[k]);
            }
        }
        fprintf(fp, "\n");
    }
}



static void print_adapt_lr (
        FILE *fp,
        long epoch,
        double adapt_lr)
{
    fprintf(fp, "%ld\t%f\n", epoch, adapt_lr);
}


static void print_mre_error (
        FILE *fp,
        long epoch,
        const struct mixture_of_rnn_experts *mre)
{
    double error[mre->series_num];
    double joint_likelihood[mre->series_num];
    double total_likelihood[mre->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        const struct mre_state *mre_s = mre->mre_s + i;
        error[i] = mre_get_error(mre_s);
        joint_likelihood[i] = mre_get_joint_likelihood(mre_s);
        total_likelihood[i] = joint_likelihood[i] +
            mre_get_prior_likelihood(mre_s);
        error[i] /= mre_s->length * mre->out_state_size;
        joint_likelihood[i] /= mre_s->length;
        total_likelihood[i] /= mre_s->length;
    }
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < mre->series_num; i++) {
        fprintf(fp, "\t%g\t%g\t%g", error[i], joint_likelihood[i],
                total_likelihood[i]);
    }
    fprintf(fp, "\n");
}


static void print_mre_state (
        FILE *fp,
        const struct mre_state *mre_s)
{
    if (mre_s->mre->expert_num <= 0) return;
    const struct rnn_state *rnn_s = mre_s->expert_rnn_s[0];
    for (int n = 0; n < mre_s->length; n++) {
        fprintf(fp, "%d", n);
        for (int i = 0; i < mre_s->mre->out_state_size; i++) {
            fprintf(fp, "\t%f\t%f", rnn_s->teach_state[n][i],
                    mre_s->out_state[n][i]);
        }
        fprintf(fp, "\n");
    }
}

static void print_mre_state_forall (
        FILE **fp_array,
        const struct mixture_of_rnn_experts *mre)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        print_mre_state(fp_array[i], mre->mre_s + i);
    }
}


static void print_mre_gate (
        FILE *fp,
        const struct mre_state *mre_s)
{
    for (int n = 0; n < mre_s->length; n++) {
        fprintf(fp, "%d", n);
        for (int i = 0; i < mre_s->mre->expert_num; i++) {
            fprintf(fp, "\t%f", mre_s->gate[i][n]);
        }
        fprintf(fp, "\n");
    }
}

static void print_mre_gate_forall (
        FILE **fp_array,
        const struct mixture_of_rnn_experts *mre)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        print_mre_gate(fp_array[i], mre->mre_s + i);
    }
}



static int enable_print (
        long epoch,
        const struct print_interval *pi)
{
    long interval;
    if (pi->use_logscale_interval) {
        interval = 1;
        while (epoch >= 10 * interval) {
            interval *= 10;
        }
        if (interval > pi->interval) {
            interval = pi->interval;
        }
    } else {
        interval = pi->interval;
    }
    return ((epoch % interval) == 0 && epoch >= pi->init && epoch <= pi->end);
}


static void print_parameters_with_epoch (
        long epoch,
        const struct general_parameters *gp,
        const struct mixture_of_rnn_experts *mre,
        struct output_files *fp_list)
{
    if (fp_list->fp_wweight &&
            enable_print(epoch, &gp->iop.interval_for_weight_file)) {
        print_mre_weight(fp_list->fp_wweight, epoch, mre);
    }

    if (fp_list->fp_wthreshold &&
            enable_print(epoch, &gp->iop.interval_for_threshold_file)) {
        print_mre_threshold(fp_list->fp_wthreshold, epoch, mre);
    }

    if (fp_list->fp_wtau &&
            enable_print(epoch, &gp->iop.interval_for_tau_file)) {
        print_mre_tau(fp_list->fp_wtau, epoch, mre);
    }

    if (fp_list->fp_wsigma &&
            enable_print(epoch, &gp->iop.interval_for_sigma_file)) {
        print_mre_sigma(fp_list->fp_wsigma, epoch, mre);
        fflush(fp_list->fp_wsigma);
    }

    if (fp_list->fp_winit &&
            enable_print(epoch, &gp->iop.interval_for_init_file)) {
        print_mre_init(fp_list->fp_winit, epoch, mre);
    }

    if (fp_list->fp_wadapt_lr &&
            enable_print(epoch, &gp->iop.interval_for_adapt_lr_file)) {
        print_adapt_lr(fp_list->fp_wadapt_lr, epoch, gp->inp.adapt_lr);
        fflush(fp_list->fp_wadapt_lr);
    }

    if (fp_list->fp_wgate_array &&
            enable_print(epoch, &gp->iop.interval_for_gate_file)) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wgate_array[i], "# epoch = %ld\n", epoch);
            fprintf(fp_list->fp_wgate_array[i], "# target:%d\n", i);
        }
        print_mre_gate_forall(fp_list->fp_wgate_array, mre);
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wgate_array[i], "\n");
        }
    }
}


static void print_open_loop_data_with_epoch (
        long epoch,
        const struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre,
        struct output_files *fp_list)
{
    int compute_forward_dynamics = 0;

    if (fp_list->fp_werror &&
            enable_print(epoch, &gp->iop.interval_for_error_file)) {
        if (!compute_forward_dynamics) {
            mre_forward_dynamics_forall(mre);
            compute_forward_dynamics = 1;
        }
        print_mre_error(fp_list->fp_werror, epoch, mre);
        fflush(fp_list->fp_werror);
    }

    if (fp_list->fp_wstate_array &&
            enable_print(epoch, &gp->iop.interval_for_state_file)) {
        if (!compute_forward_dynamics) {
            mre_forward_dynamics_forall(mre);
            compute_forward_dynamics = 1;
        }
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wstate_array[i], "# epoch = %ld\n", epoch);
            fprintf(fp_list->fp_wstate_array[i], "# target:%d\n", i);
        }
        print_mre_state_forall(fp_list->fp_wstate_array, mre);
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wstate_array[i], "\n");
        }
    }
}

static void print_closed_loop_data_with_epoch (
        long epoch,
        const struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre,
        struct output_files *fp_list)
{
    int compute_forward_dynamics = 0;

    if (fp_list->fp_wclosed_error &&
            enable_print(epoch, &gp->iop.interval_for_closed_error_file)) {
        if (!compute_forward_dynamics) {
            mre_forward_dynamics_in_closed_loop_forall(mre,
                    gp->mp.delay_length);
            compute_forward_dynamics = 1;
        }
        print_mre_error(fp_list->fp_wclosed_error, epoch, mre);
        fflush(fp_list->fp_wclosed_error);
    }

    if (fp_list->fp_wclosed_state_array &&
            enable_print(epoch, &gp->iop.interval_for_closed_state_file)) {
        if (!compute_forward_dynamics) {
            mre_forward_dynamics_in_closed_loop_forall(mre,
                    gp->mp.delay_length);
            compute_forward_dynamics = 1;
        }
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wclosed_state_array[i], "# epoch = %ld\n",
                    epoch);
            fprintf(fp_list->fp_wclosed_state_array[i],
                    "# target:%d (closed loop)\n", i);
        }
        print_mre_state_forall(fp_list->fp_wclosed_state_array, mre);
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wclosed_state_array[i], "\n");
        }
    }
}


void print_training_main_begin (
        const struct general_parameters *gp,
        const struct mixture_of_rnn_experts *mre,
        struct output_files *fp_list)
{
    if (fp_list->fp_wstate_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wstate_array[i], "# MRE STATE FILE\n");
            print_general_parameters(fp_list->fp_wstate_array[i], gp);
            print_mre_parameters(fp_list->fp_wstate_array[i], mre);
        }
    }
    if (fp_list->fp_wclosed_state_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wclosed_state_array[i],  "# MRE STATE FILE\n");
            print_general_parameters(fp_list->fp_wclosed_state_array[i], gp);
            print_mre_parameters(fp_list->fp_wclosed_state_array[i], mre);
        }
    }
    if (fp_list->fp_wgate_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wgate_array[i], "# MRE GATE FILE\n");
            print_general_parameters(fp_list->fp_wgate_array[i], gp);
            print_mre_parameters(fp_list->fp_wgate_array[i], mre);
        }
    }
    if (fp_list->fp_wweight) {
        fprintf(fp_list->fp_wweight, "# MRE WEIGHT FILE\n");
        print_general_parameters(fp_list->fp_wweight, gp);
        print_mre_parameters(fp_list->fp_wweight, mre);
    }
    if (fp_list->fp_wthreshold) {
        fprintf(fp_list->fp_wthreshold, "# MRE THRESHOLD FILE\n");
        print_general_parameters(fp_list->fp_wthreshold, gp);
        print_mre_parameters(fp_list->fp_wthreshold, mre);
    }
    if (fp_list->fp_wtau) {
        fprintf(fp_list->fp_wtau, "# MRE TAU FILE\n");
        print_general_parameters(fp_list->fp_wtau, gp);
        print_mre_parameters(fp_list->fp_wtau, mre);
    }
    if (fp_list->fp_wsigma) {
        fprintf(fp_list->fp_wsigma, "# MRE SIGMA FILE\n");
        print_general_parameters(fp_list->fp_wsigma, gp);
        print_mre_parameters(fp_list->fp_wsigma, mre);
    }
    if (fp_list->fp_winit) {
        fprintf(fp_list->fp_winit, "# MRE INIT FILE\n");
        print_general_parameters(fp_list->fp_winit, gp);
        print_mre_parameters(fp_list->fp_winit, mre);
    }
    if (fp_list->fp_wadapt_lr) {
        fprintf(fp_list->fp_wadapt_lr, "# MRE ADAPT_LR FILE\n");
        print_general_parameters(fp_list->fp_wadapt_lr, gp);
        print_mre_parameters(fp_list->fp_wadapt_lr, mre);
    }
    if (fp_list->fp_werror) {
        fprintf(fp_list->fp_werror, "# MRE ERROR FILE\n");
        print_general_parameters(fp_list->fp_werror, gp);
        print_mre_parameters(fp_list->fp_werror, mre);
    }
    if (fp_list->fp_wclosed_error) {
        fprintf(fp_list->fp_wclosed_error, "# MRE ERROR FILE\n");
        print_general_parameters(fp_list->fp_wclosed_error, gp);
        print_mre_parameters(fp_list->fp_wclosed_error, mre);
    }
}


void print_training_main_loop (
        long epoch,
        const struct general_parameters *gp,
        struct mixture_of_rnn_experts *mre,
        struct output_files *fp_list)
{
    print_parameters_with_epoch(epoch, gp, mre, fp_list);
    print_open_loop_data_with_epoch(epoch, gp, mre, fp_list);
    print_closed_loop_data_with_epoch(epoch, gp, mre, fp_list);
}


