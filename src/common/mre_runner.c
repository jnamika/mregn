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
#include <assert.h>
#include "mt19937ar.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "mre_runner.h"


/******************************************************************************/
/********** Initialization and Free *******************************************/
/******************************************************************************/

int _new_mre_runner (struct mre_runner **runner)
{
    (*runner) = malloc(sizeof(struct mre_runner));
    if ((*runner) == NULL) {
        return 1;
    }
    return 0;
}

void _delete_mre_runner (struct mre_runner *runner)
{
    free(runner);
}



void init_mre_runner (
        struct mre_runner *runner,
        FILE *mre_fp,
        FILE *gn_fp)
{
    int delay_length;

    FREAD(&delay_length, 1, mre_fp);
    fread_mixture_of_rnn_experts(&runner->mre, mre_fp);
    mre_add_target(&runner->mre, delay_length, NULL, NULL);

    FREAD(&delay_length, 1, gn_fp);
    fread_recurrent_neural_network(&runner->gn, gn_fp);
    rnn_add_target(&runner->gn, delay_length, NULL, NULL);

    runner->id = runner->gn.series_num - 1;
}


void free_mre_runner (struct mre_runner *runner)
{
    free_mixture_of_rnn_experts(&runner->mre);
    free_recurrent_neural_network(&runner->gn);
}


static void copy_init_state (
        struct rnn_state *dst,
        struct rnn_state *src)
{
    for (int n = 0; n < dst->length; n++) {
        if (n < src->length) {
            memmove(dst->in_state[n], src->in_state[n], sizeof(double) *
                    dst->rnn_p->in_state_size);
        } else {
            for (int i = 0; i < dst->rnn_p->in_state_size; i++) {
                dst->in_state[n][i] = (2*genrand_real3()-1);
            }
        }
    }
    memmove(dst->init_c_state, src->init_c_state, sizeof(double) *
            dst->rnn_p->c_state_size);
    memmove(dst->init_c_inter_state, src->init_c_inter_state, sizeof(double) *
            dst->rnn_p->c_state_size);
}

static void random_init_state (struct rnn_state *rnn_s)
{
    for (int n = 0; n < rnn_s->length; n++) {
        for (int i = 0; i < rnn_s->rnn_p->in_state_size; i++) {
            rnn_s->in_state[n][i] = (2*genrand_real3()-1);
        }
    }
    for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
        rnn_s->init_c_state[i] = (2*genrand_real3()-1);
        rnn_s->init_c_inter_state[i] = atanh(rnn_s->init_c_state[i]);
    }
}


void set_init_state_of_mre_runner (
        struct mre_runner *runner,
        int series_id)
{
    if (series_id >= 0 && series_id < runner->id) {
        copy_init_state(runner->gn.rnn_s + runner->id,
                runner->gn.rnn_s + series_id);
        for (int i = 0; i < runner->mre.expert_num; i++) {
            copy_init_state(runner->mre.expert_rnn[i].rnn_s + runner->id,
                    runner->mre.expert_rnn[i].rnn_s + series_id);
        }
    } else {
        random_init_state(runner->gn.rnn_s + runner->id);
        for (int i = 0; i < runner->mre.expert_num; i++) {
            random_init_state(runner->mre.expert_rnn[i].rnn_s +
                    runner->id);
        }
    }
}


/******************************************************************************/
/********** Computation of forward dynamics ***********************************/
/******************************************************************************/

#define IN_STATE(X) ((X)->expert_rnn_s[0]->in_state[0])

static void mregn_fmap (
        struct rnn_state *gn_s,
        struct mre_state *mre_s)
{
    const int expert_num = mre_s->mre->expert_num;
    const int in_state_size = mre_s->mre->in_state_size;
    const int out_state_size = mre_s->mre->out_state_size;
    double in_state[expert_num + in_state_size];

    assert(mre_s->mre->in_state_size <= mre_s->mre->out_state_size);
    assert(mre_s->mre->expert_num == gn_s->rnn_p->out_state_size);
    assert(gn_s->rnn_p->in_state_size == gn_s->rnn_p->out_state_size +
            mre_s->mre->in_state_size);

    memcpy(in_state, gn_s->in_state[0], sizeof(double) * expert_num);
    memcpy(in_state + expert_num, IN_STATE(mre_s), sizeof(double) *
            in_state_size);

    for (int i = 0; i < expert_num; i++) {
        struct rnn_state *rnn_s = mre_s->expert_rnn_s[i];
        rnn_forward_map(rnn_s->rnn_p, in_state + expert_num,
                rnn_s->init_c_inter_state, rnn_s->init_c_state,
                rnn_s->c_inputsum[0], rnn_s->c_inter_state[0],
                rnn_s->c_state[0], rnn_s->o_inter_state[0],
                rnn_s->out_state[0]);
    }
    rnn_forward_map(gn_s->rnn_p, in_state, gn_s->init_c_inter_state,
            gn_s->init_c_state, gn_s->c_inputsum[0], gn_s->c_inter_state[0],
            gn_s->c_state[0], gn_s->o_inter_state[0], gn_s->out_state[0]);
    for (int i = 0; i < out_state_size; i++) {
        mre_s->out_state[0][i] = 0;
        for (int j = 0; j < expert_num; j++) {
            mre_s->out_state[0][i] += in_state[j] *
                mre_s->expert_rnn_s[j]->out_state[0][i];
        }
    }

    for (int k = 1; k < gn_s->length; k++) {
        memmove(gn_s->in_state[k-1], gn_s->in_state[k], sizeof(double) *
                expert_num);
    }
    memmove(gn_s->in_state[gn_s->length-1], gn_s->out_state[0], sizeof(double) *
            expert_num);
    memmove(gn_s->in_state[gn_s->length-1] + expert_num, mre_s->out_state[0],
            sizeof(double) * in_state_size);
    for (int i = 0; i < expert_num; i++) {
        struct rnn_state *rnn_s = mre_s->expert_rnn_s[i];
        for (int k = 1; k < mre_s->length; k++) {
            memmove(rnn_s->in_state[k-1], rnn_s->in_state[k], sizeof(double) *
                    in_state_size);
        }
        memmove(rnn_s->in_state[mre_s->length-1], mre_s->out_state[0],
                sizeof(double) * in_state_size);
    }

    memmove(gn_s->init_c_state, gn_s->c_state[0], sizeof(double) *
            gn_s->rnn_p->c_state_size);
    memmove(gn_s->init_c_inter_state, gn_s->c_inter_state[0],
            sizeof(double) * gn_s->rnn_p->c_state_size);

    for (int i = 0; i < expert_num; i++) {
        struct rnn_state *rnn_s = mre_s->expert_rnn_s[i];
        memmove(rnn_s->init_c_state, rnn_s->c_state[0], sizeof(double) *
                rnn_s->rnn_p->c_state_size);
        memmove(rnn_s->init_c_inter_state, rnn_s->c_inter_state[0],
                sizeof(double) * rnn_s->rnn_p->c_state_size);
    }
}


void update_mre_runner (struct mre_runner *runner)
{
    mregn_fmap(runner->gn.rnn_s + runner->id, runner->mre.mre_s + runner->id);
}



/******************************************************************************/
/********** Interface *********************************************************/
/******************************************************************************/

int gn_in_state_size_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_p.in_state_size;
}

int gn_c_state_size_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_p.c_state_size;
}

int gn_out_state_size_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_p.out_state_size;
}

int gn_delay_length_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_s[runner->id].length;
}

int gn_target_num_from_runner (struct mre_runner *runner)
{
    return runner->id;
}

double* gn_in_state_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_s[runner->id].in_state[0];
}

double* gn_c_state_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_s[runner->id].init_c_state;
}

double* gn_c_inter_state_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_s[runner->id].init_c_inter_state;
}

double* gn_out_state_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_s[runner->id].out_state[0];
}

struct rnn_state* gn_state_from_runner (struct mre_runner *runner)
{
    return runner->gn.rnn_s + runner->id;
}


int mre_in_state_size_from_runner (struct mre_runner *runner)
{
    return runner->mre.in_state_size;
}

int mre_out_state_size_from_runner (struct mre_runner *runner)
{
    return runner->mre.out_state_size;
}

int mre_delay_length_from_runner (struct mre_runner *runner)
{
    return runner->mre.mre_s[runner->id].length;
}

int mre_target_num_from_runner (struct mre_runner *runner)
{
    return runner->id;
}

double* mre_in_state_from_runner (struct mre_runner *runner)
{
    return IN_STATE(runner->mre.mre_s + runner->id);
}

double* mre_out_state_from_runner (struct mre_runner *runner)
{
    return runner->mre.mre_s[runner->id].out_state[0];
}

struct mre_state* mre_state_from_runner (struct mre_runner *runner)
{
    return runner->mre.mre_s + runner->id;
}


int expert_rnn_c_state_size_from_runner (
        struct mre_runner *runner,
        int index)
{
    if (index < 0 || index >= runner->mre.expert_num) {
        return 0;
    } else {
        return runner->mre.expert_rnn[index].rnn_p.c_state_size;
    }
}

double* expert_rnn_c_state_from_runner (
        struct mre_runner *runner,
        int index)
{
    if (index < 0 || index >= runner->mre.expert_num) {
        return NULL;
    } else {
        struct rnn_state *rnn_s;
        rnn_s = runner->mre.mre_s[runner->id].expert_rnn_s[index];
        return rnn_s->init_c_state;
    }
}

double* expert_rnn_c_inter_state_from_runner (
        struct mre_runner *runner,
        int index)
{
    if (index < 0 || index >= runner->mre.expert_num) {
        return NULL;
    } else {
        struct rnn_state *rnn_s;
        rnn_s = runner->mre.mre_s[runner->id].expert_rnn_s[index];
        return rnn_s->init_c_inter_state;
    }
}

struct rnn_state* expert_rnn_state_from_runner (
        struct mre_runner *runner,
        int index)
{
    return runner->mre.mre_s[runner->id].expert_rnn_s[index];
}

