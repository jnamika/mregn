/*
    Copyright (c) 2011, Jun Namikawa <jnamika@gmail.com>

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
#include <math.h>
#include <string.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "lyapunov.h"
#include "mregn_runner.h"
#include "mregn_lyapunov.h"


static void memcopy_rnn_state (
        struct rnn_state *dst,
        struct rnn_state *src,
        int dst_step,
        int src_step)
{
    const struct rnn_parameters *rnn_p = src->rnn_p;
    memcpy(dst->in_state[dst_step], src->in_state[src_step],
            rnn_p->in_state_size * sizeof(double));
    memcpy(dst->c_inputsum[dst_step], src->c_inputsum[src_step],
            rnn_p->c_state_size * sizeof(double));
    memcpy(dst->c_inter_state[dst_step], src->c_inter_state[src_step],
            rnn_p->c_state_size * sizeof(double));
    memcpy(dst->c_state[dst_step], src->c_state[src_step],
            rnn_p->c_state_size * sizeof(double));
    memcpy(dst->o_inter_state[dst_step], src->o_inter_state[src_step],
            rnn_p->out_state_size * sizeof(double));
    memcpy(dst->out_state[dst_step], src->out_state[src_step],
            rnn_p->out_state_size * sizeof(double));
    if (dst_step == 0 && src_step == 0) {
        memcpy(dst->init_c_inter_state, src->init_c_inter_state,
                rnn_p->c_state_size * sizeof(double));
        memcpy(dst->init_c_state, src->init_c_state,
                rnn_p->c_state_size * sizeof(double));
    }
}

static void memcopy_mregn_state (
        struct mre_state *dst_mre_s,
        struct rnn_state *dst_gn_s,
        struct mre_state *src_mre_s,
        struct rnn_state *src_gn_s,
        int dst_step,
        int src_step)
{
    memcopy_rnn_state(dst_gn_s, src_gn_s, dst_step, src_step);
    for (int i = 0; i < src_mre_s->mre->expert_num; i++) {
        memcopy_rnn_state(dst_mre_s->expert_rnn_s[i],
                src_mre_s->expert_rnn_s[i], dst_step, src_step);
    }
    for (int i = 0; i < src_mre_s->mre->expert_num; i++) {
        dst_mre_s->gate[i][dst_step] = src_mre_s->gate[i][src_step];
    }
    for (int i = 0; i < src_mre_s->mre->out_state_size; i++) {
        dst_mre_s->out_state[dst_step][i] = src_mre_s->out_state[src_step][i];
    }
}


static double gauss_dev()
{
    static int iset = 0;
    static double gset;
    double fac, rsq, v1, v2;
    if (iset == 0) {
        do {
            v1 = 2.0 * genrand_real1() - 1.0;
            v2 = 2.0 * genrand_real1() - 1.0;
            rsq = v1*v1 + v2*v2;
        } while (rsq >= 1.0 || fpclassify(rsq) == FP_ZERO);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        iset = 1;
        return v2 * fac;
    } else {
        iset = 0;
        return gset;
    }
}

static void update_mregn_runner_with_noise (
        struct mregn_runner *runner,
        double noise_deviation)
{
    double *in_state = mre_in_state_from_runner(runner);
    if (fpclassify(noise_deviation) != FP_ZERO) {
        for (int i = 0; i < runner->mre.in_state_size; i++) {
            in_state[i] += noise_deviation * gauss_dev();
        }
    }
    update_mregn_runner(runner);

}

void compute_lyapunov_main (
        const struct analysis_parameters *ap,
        struct mregn_runner *runner)
{
    mre_add_target(&runner->mre, ap->mem_size, NULL, NULL);
    rnn_add_target(&runner->gn, ap->mem_size, NULL, NULL);
    const int mre_delay_length = mre_delay_length_from_runner(runner);
    const int gn_delay_length = gn_delay_length_from_runner(runner);
    struct mre_state *mre_s = runner->mre.mre_s + runner->mre.series_num - 1;
    struct rnn_state *gn_s = runner->gn.rnn_s + runner->gn.series_num - 1;
    struct mregn_lyapunov_info rl_info;
    init_mregn_lyapunov_info(&rl_info, mre_s, gn_s, mre_delay_length,
            gn_delay_length, 0);

    int spectrum_size;
    if (ap->lyapunov_spectrum_size < 0 ||
            ap->lyapunov_spectrum_size > rl_info.dimension) {
        spectrum_size = rl_info.dimension;
    } else {
        spectrum_size = ap->lyapunov_spectrum_size;
    }
    double lyapunov[spectrum_size], tmp[spectrum_size];
    for (int i = 0; i < ap->sample_num; i++) {
        set_init_state_of_mregn_runner(runner, -1);
        for (long n = 0; n < ap->truncate_length; n++) {
            update_mregn_runner_with_noise(runner, ap->noise_deviation);
        }
        for (int j = 0; j < spectrum_size; j++) {
            lyapunov[j] = 0;
        }
        for (long n = 0; n < ap->length; n++) {
            int m = (int)(n % gn_s->length);
            update_mregn_runner_with_noise(runner, ap->noise_deviation);
            memcopy_mregn_state(mre_s, gn_s, mre_state_from_runner(runner),
                    gn_state_from_runner(runner), m, 0);
            if ((m+1) >= gn_s->length) {
                mregn_lyapunov_spectrum(&rl_info, tmp, spectrum_size);
                for (int j = 0; j < spectrum_size; j++) {
                    lyapunov[j] += tmp[j] * gn_s->length;
                }
            }
        }
        if ((ap->length % gn_s->length) > 0) {
            int len = gn_s->length;
            gn_s->length = ap->length % gn_s->length;
            mre_s->length = gn_s->length;
            mregn_lyapunov_spectrum(&rl_info, tmp, spectrum_size);
            for (int j = 0; j < spectrum_size; j++) {
                lyapunov[j] += tmp[j] * gn_s->length;
            }
            gn_s->length = len;
            mre_s->length = len;
        }
        for (int j = 0; j < spectrum_size; j++) {
            printf("%f%c", lyapunov[j] / ap->length,
                    (j + 1 < spectrum_size) ? '\t' : '\n');
        }
    }
    free_mregn_lyapunov_info(&rl_info);
}

