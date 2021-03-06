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

#ifndef MREGN_LYAPUNOV_H
#define MREGN_LYAPUNOV_H

#include "rnn.h"
#include "mre.h"


typedef struct mregn_lyapunov_info {
    const struct mre_state *mre_s;
    const struct rnn_state *gn_s;
    int mre_delay_length;
    int gn_delay_length;
    int truncate_length;
    int expert_num;

    int dimension;
    double ***tmp_mre_matrix;
    double **tmp_gn_matrix;

    int length;
    double **state;
} mregn_lyapunov_info;


void init_mregn_lyapunov_info (
        struct mregn_lyapunov_info *ml_info,
        const struct mre_state *mre_s,
        const struct rnn_state *gn_s,
        int mre_delay_length,
        int gn_delay_length,
        int truncate_length);

void mregn_lyapunov_info_alloc (struct mregn_lyapunov_info *ml_info);

void free_mregn_lyapunov_info (struct mregn_lyapunov_info *ml_info);

void reset_mregn_lyapunov_info (struct mregn_lyapunov_info *ml_info);

double* mregn_lyapunov_spectrum (
        struct mregn_lyapunov_info *ml_info,
        double *spectrum,
        int spectrum_size);

#endif

