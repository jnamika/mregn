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

#ifndef MRE_RUNNER_H
#define MRE_RUNNER_H

#include "mre.h"

typedef struct mre_runner {
    int id;
    struct mixture_of_rnn_experts mre;
    struct recurrent_neural_network gn;
} mre_runner;



int _new_mre_runner (struct mre_runner **runner);

void _delete_mre_runner (struct mre_runner *runner);


void init_mre_runner (
        struct mre_runner *runner,
        FILE *mre_fp,
        FILE *gn_fp);

void free_mre_runner (struct mre_runner *runner);

void set_init_state_of_mre_runner (
        struct mre_runner *runner,
        int series_id);

void update_mre_runner (struct mre_runner *runner);


int gn_in_state_size_from_runner (struct mre_runner *runner);
int gn_c_state_size_from_runner (struct mre_runner *runner);
int gn_out_state_size_from_runner (struct mre_runner *runner);
int gn_delay_length_from_runner (struct mre_runner *runner);
int gn_target_num_from_runner (struct mre_runner *runner);
double* gn_in_state_from_runner (struct mre_runner *runner);
double* gn_c_state_from_runner (struct mre_runner *runner);
double* gn_c_inter_state_from_runner (struct mre_runner *runner);
double* gn_out_state_from_runner (struct mre_runner *runner);
struct rnn_state* gn_state_from_runner (struct mre_runner *runner);

int mre_in_state_size_from_runner (struct mre_runner *runner);
int mre_out_state_size_from_runner (struct mre_runner *runner);
int mre_delay_length_from_runner (struct mre_runner *runner);
int mre_target_num_from_runner (struct mre_runner *runner);
double* mre_in_state_from_runner (struct mre_runner *runner);
double* mre_out_state_from_runner (struct mre_runner *runner);
struct mre_state* mre_state_from_runner (struct mre_runner *runner);

int expert_rnn_c_state_size_from_runner (
        struct mre_runner *runner,
        int index);
double* expert_rnn_c_state_from_runner (
        struct mre_runner *runner,
        int index);
double* expert_rnn_c_inter_state_from_runner (
        struct mre_runner *runner,
        int index);
struct rnn_state* expert_rnn_state_from_runner (
        struct mre_runner *runner,
        int index);

#endif

