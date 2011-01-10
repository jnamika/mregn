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
#include "mregn.h"


static inline void copy_gn_output_to_gate (
        struct mre_state *mre_s,
        struct rnn_state *gn_s,
        int gn_delay_length)
{
    int length;

    length = gn_s->length + gn_delay_length;
    if (length > mre_s->length) length = mre_s->length;
    for (int i = 0; i < mre_s->mre->expert_num; i++) {
        for (int n = gn_delay_length; n < length; n++) {
            mre_s->gate[i][n] = gn_s->out_state[n-gn_delay_length][i];
        }
    }
}

void mregn_forward_dynamics (
        struct mre_state *mre_s,
        struct rnn_state *gn_s,
        int gn_delay_length)
{
    rnn_forward_dynamics(gn_s);
    copy_gn_output_to_gate(mre_s, gn_s, gn_delay_length);
    mre_forward_dynamics(mre_s);
}


void mregn_forward_dynamics_forall (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int gn_delay_length)
{
    rnn_forward_dynamics_forall(gn);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        copy_gn_output_to_gate(mre->mre_s + i, gn->rnn_s + i, gn_delay_length);
    }
    mre_forward_dynamics_forall(mre);
}




void mregn_forward_dynamics_in_closed_loop (
        struct mre_state *mre_s,
        struct rnn_state *gn_s,
        int mre_delay_length,
        int gn_delay_length)
{
    const struct rnn_parameters *gn_p = gn_s->rnn_p;
    double in_state[gn_p->in_state_size];

    assert(mre_s->mre->in_state_size <= mre_s->mre->out_state_size);
    assert(mre_s->mre->expert_num == gn_p->out_state_size);
    assert(gn_p->in_state_size == gn_p->out_state_size +
            mre_s->mre->in_state_size);

    for (int n = 0; n < gn_s->length && n < mre_s->length; n++) {
        struct rnn_state *rnn_s;
        if (n == 0) {
            for (int i = 0; i < mre_s->mre->expert_num; i++) {
                rnn_s = mre_s->expert_rnn_s[i];
                rnn_forward_map(rnn_s->rnn_p, rnn_s->in_state[0],
                        rnn_s->init_c_inter_state, rnn_s->init_c_state,
                        rnn_s->c_inputsum[0], rnn_s->c_inter_state[0],
                        rnn_s->c_state[0], rnn_s->o_inter_state[0],
                        rnn_s->out_state[0]);
            }
        } else if (n < mre_delay_length) {
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
                rnn_forward_map(rnn_s->rnn_p,
                        mre_s->out_state[n-mre_delay_length],
                        rnn_s->c_inter_state[n-1], rnn_s->c_state[n-1],
                        rnn_s->c_inputsum[n], rnn_s->c_inter_state[n],
                        rnn_s->c_state[n], rnn_s->o_inter_state[n],
                        rnn_s->out_state[n]);
            }
        }

        if (n >= gn_delay_length) {
            for (int i = 0; i < mre_s->mre->expert_num; i++) {
                mre_s->gate[i][n] = gn_s->out_state[n-gn_delay_length][i];
            }
        }
        for (int i = 0; i < mre_s->mre->out_state_size; i++) {
            mre_s->out_state[n][i] = 0;
            for (int j = 0; j < mre_s->mre->expert_num; j++) {
                mre_s->out_state[n][i] += mre_s->gate[j][n] *
                    mre_s->expert_rnn_s[j]->out_state[n][i];
            }
        }

        if (n == 0) {
            rnn_forward_map(gn_p, gn_s->in_state[0], gn_s->init_c_inter_state,
                    gn_s->init_c_state, gn_s->c_inputsum[0],
                    gn_s->c_inter_state[0], gn_s->c_state[0],
                    gn_s->o_inter_state[0], gn_s->out_state[0]);
        } else {
            if (n < gn_delay_length) {
                memcpy(in_state, gn_s->in_state[n], sizeof(double) *
                        gn_p->out_state_size);
            } else {
                memcpy(in_state, gn_s->out_state[n-gn_delay_length],
                        sizeof(double) * gn_p->out_state_size);
            }
            if (n < mre_delay_length) {
                memcpy(in_state + gn_p->out_state_size, gn_s->in_state[n] +
                        gn_p->out_state_size, sizeof(double) *
                        mre_s->mre->in_state_size);
            } else {
                memcpy(in_state + gn_p->out_state_size,
                        mre_s->out_state[n-mre_delay_length], sizeof(double) *
                        mre_s->mre->in_state_size);
            }
            rnn_forward_map(gn_p, in_state, gn_s->c_inter_state[n-1],
                    gn_s->c_state[n-1], gn_s->c_inputsum[n],
                    gn_s->c_inter_state[n], gn_s->c_state[n],
                    gn_s->o_inter_state[n], gn_s->out_state[n]);
        }
    }
}



void mregn_forward_dynamics_in_closed_loop_forall (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int mre_delay_length,
        int gn_delay_length)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mre->series_num; i++) {
        mregn_forward_dynamics_in_closed_loop(mre->mre_s + i, gn->rnn_s + i,
                mre_delay_length, gn_delay_length);
    }
}


