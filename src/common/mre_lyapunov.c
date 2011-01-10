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
#include <math.h>
#include <assert.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "solver.h"
#include "mre_lyapunov.h"




void init_mre_lyapunov_info (
        struct mre_lyapunov_info *ml_info,
        const struct mre_state *mre_s,
        const struct rnn_state *gn_s,
        int mre_delay_length,
        int gn_delay_length,
        int truncate_length)
{
    const int min_length = (gn_s->length < mre_s->length) ? gn_s->length :
        mre_s->length;

    assert(mre_s->mre->in_state_size == mre_s->mre->out_state_size ||
            mre_s->mre->in_state_size == 0);
    assert(mre_s->mre->expert_num == gn_s->rnn_p->out_state_size);
    assert(gn_s->rnn_p->in_state_size == gn_s->rnn_p->out_state_size +
            mre_s->mre->in_state_size);
    assert(gn_delay_length > 0);
    assert(mre_delay_length > 0);
    assert(truncate_length >= 0);
    assert(min_length > truncate_length);

    ml_info->gn_s = gn_s;
    ml_info->mre_s = mre_s;
    ml_info->gn_delay_length = gn_delay_length;
    ml_info->mre_delay_length = mre_delay_length;
    ml_info->truncate_length = truncate_length;
    ml_info->expert_num = mre_s->mre->expert_num;
    ml_info->length = min_length - truncate_length;
    ml_info->dimension = mre_s->mre->in_state_size * mre_delay_length;
    ml_info->dimension += gn_s->rnn_p->out_state_size * gn_delay_length;
    for (int i = 0; i < mre_s->mre->expert_num; i++) {
        ml_info->dimension += mre_s->expert_rnn_s[i]->rnn_p->c_state_size;
    }
    ml_info->dimension += gn_s->rnn_p->c_state_size;

    mre_lyapunov_info_alloc(ml_info);
}

void mre_lyapunov_info_alloc (struct mre_lyapunov_info *ml_info)
{
    const struct rnn_state *gn_s = ml_info->gn_s;
    const struct mre_state *mre_s = ml_info->mre_s;
    const int tmp_in = gn_s->rnn_p->in_state_size + gn_s->rnn_p->c_state_size;
    const int tmp_out = gn_s->rnn_p->out_state_size + gn_s->rnn_p->c_state_size;
    MALLOC(ml_info->tmp_gn_matrix, tmp_out);
    MALLOC(ml_info->tmp_gn_matrix[0], tmp_out * tmp_in);
    for (int i = 1; i < tmp_out ; i++) {
        ml_info->tmp_gn_matrix[i] = ml_info->tmp_gn_matrix[0] + i * tmp_in;
    }

    MALLOC(ml_info->tmp_mre_matrix, mre_s->mre->expert_num);
    for (int i = 0; i < mre_s->mre->expert_num; i++) {
        int tmp_dimension = mre_s->expert_rnn_s[i]->rnn_p->out_state_size +
            mre_s->expert_rnn_s[i]->rnn_p->c_state_size;
        MALLOC(ml_info->tmp_mre_matrix[i], tmp_dimension);
        MALLOC(ml_info->tmp_mre_matrix[i][0], tmp_dimension * tmp_dimension);
        for (int j = 1; j < tmp_dimension; j++) {
            ml_info->tmp_mre_matrix[i][j] = ml_info->tmp_mre_matrix[i][0] + j *
                tmp_dimension;
        }
    }

    MALLOC(ml_info->state, ml_info->length);
    MALLOC(ml_info->state[0], ml_info->length * ml_info->dimension);
    for (int i = 1; i < ml_info->length; i++) {
        ml_info->state[i] = ml_info->state[0] + i * ml_info->dimension;
    }
}

void free_mre_lyapunov_info (struct mre_lyapunov_info *ml_info)
{
    free(ml_info->tmp_gn_matrix[0]);
    free(ml_info->tmp_gn_matrix);
    for (int i = 0; i < ml_info->expert_num; i++) {
        free(ml_info->tmp_mre_matrix[i][0]);
        free(ml_info->tmp_mre_matrix[i]);
    }
    free(ml_info->tmp_mre_matrix);

    free(ml_info->state[0]);
    free(ml_info->state);
}


static double** jacobian_matrix_with_delay (
        double** matrix,
        double*** mre_tmp_matrix,
        double** gn_tmp_matrix,
        int dimension,
        const struct mre_state *mre_s,
        const struct rnn_state *gn_s,
        int mre_delay_length,
        int gn_delay_length,
        int T)
{
    int I, J;
    const int io_size = mre_s->mre->out_state_size;
    const int expert_num = mre_s->mre->expert_num;
    const struct rnn_state *rnn_s;

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            matrix[i][j] = 0.0;
        }
    }

    rnn_jacobian_matrix(gn_tmp_matrix, gn_s->rnn_p, (T==0) ? gn_s->init_c_state
            : gn_s->c_state[T-1], gn_s->c_state[T], gn_s->out_state[T]);

    for (int i = 0; i < expert_num; i++) {
        rnn_s = mre_s->expert_rnn_s[i];
        rnn_jacobian_matrix(mre_tmp_matrix[i], rnn_s->rnn_p, (T==0) ?
                rnn_s->init_c_state : rnn_s->c_state[T-1], rnn_s->c_state[T],
                rnn_s->out_state[T]);
    }

    I = 0; // index of output at time T
    for (int i = 0; i < io_size; i++) {
        J = io_size * (mre_delay_length-1); // index of input at time T
        for (int j = 0; j < io_size; j++) {
            for (int k = 0; k < expert_num; k++) {
                matrix[I+i][J+j] += mre_s->gate[k][T] * mre_tmp_matrix[k][i][j];
            }
        }
        // index of input gate opening values
        J = io_size * mre_delay_length + expert_num * (gn_delay_length-1);
        for (int j = 0; j < expert_num; j++) {
            matrix[I+i][J+j] = mre_s->expert_rnn_s[j]->out_state[T][i];
        }
        // index of context for each expert RNN
        J = io_size * mre_delay_length + expert_num * gn_delay_length;
        for (int k = 0; k < expert_num; k++) {
            rnn_s = mre_s->expert_rnn_s[k];
            for (int j = 0; j < rnn_s->rnn_p->c_state_size; j++) {
                matrix[I+i][J+j] = mre_s->gate[k][T] *
                    mre_tmp_matrix[k][i][j+io_size];
            }
            J += rnn_s->rnn_p->c_state_size;
        }
    }
    I = io_size * mre_delay_length; // index of output gate opening values
    for (int i = 0; i < expert_num; i++) {
        J = io_size * (mre_delay_length-1); // index of input at time T
        for (int j = 0; j < io_size; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i][j + expert_num];
        }
        // index of input gate opening values
        J = io_size * mre_delay_length + expert_num * (gn_delay_length-1);
        for (int j = 0; j < expert_num; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i][j];
        }
        // index of context of the gating network
        J = dimension - gn_s->rnn_p->c_state_size;
        for (int j = 0; j < gn_s->rnn_p->c_state_size; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i][j + expert_num + io_size];
        }
    }
    // index of context for each expert RNN
    I = io_size * mre_delay_length + expert_num * gn_delay_length;
    for (int k = 0; k < expert_num; k++) {
        rnn_s = mre_s->expert_rnn_s[k];
        J = io_size * (mre_delay_length-1); // index of input at time T
        for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
            for (int j = 0; j < io_size; j++) {
                matrix[I+i][J+j] += mre_tmp_matrix[k][i + io_size][j];
            }
        }
        J = I; // index of context for each expert RNN
        for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
            for (int j = 0; j < rnn_s->rnn_p->c_state_size; j++) {
                matrix[I+i][J+j] = mre_tmp_matrix[k][i + io_size][j + io_size];
            }
        }
        I += rnn_s->rnn_p->c_state_size;
    }
    // index of context of the gating network
    I = dimension - gn_s->rnn_p->c_state_size;
    for (int i = 0; i < gn_s->rnn_p->c_state_size; i++) {
        J = io_size * (mre_delay_length-1); // index of input at time T
        for (int j = 0; j < io_size; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i + expert_num][j + expert_num];
        }
        // index of input gate opening values
        J = io_size * mre_delay_length + expert_num * (gn_delay_length-1);
        for (int j = 0; j < expert_num; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i + expert_num][j];
        }
        J = I; // index of context of the gating network
        for (int j = 0; j < gn_s->rnn_p->c_state_size; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i + expert_num][j + expert_num +
                io_size];
        }
    }

    for (int t = 1; t < mre_delay_length; t++) {
        I = io_size * t; // index of output at time T-t
        J = io_size * (t-1); // index of output at time T-t+1
        for (int i = 0; i < io_size; i++) {
            matrix[I+i][J+i] = 1;
        }
    }
    for (int t = 1; t < gn_delay_length; t++) {
        // index of output gate opening values at time T-t
        I = io_size * mre_delay_length + expert_num * t;
        // index of output gate opening values at time T-t+1
        J = io_size * mre_delay_length + expert_num * (t-1);
        for (int i = 0; i < expert_num; i++) {
            matrix[I+i][J+i] = 1;
        }
    }

    return matrix;
}

static double** jacobian_matrix_without_input (
        double** matrix,
        double*** mre_tmp_matrix,
        double** gn_tmp_matrix,
        int dimension,
        const struct mre_state *mre_s,
        const struct rnn_state *gn_s,
        int gn_delay_length,
        int T)
{
    int I, J;
    const int expert_num = mre_s->mre->expert_num;
    const struct rnn_state *rnn_s;

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            matrix[i][j] = 0.0;
        }
    }

    rnn_jacobian_matrix(gn_tmp_matrix, gn_s->rnn_p, (T==0) ?
            gn_s->init_c_state : gn_s->c_state[T-1], gn_s->c_state[T],
            gn_s->out_state[T]);

    for (int i = 0; i < expert_num; i++) {
        rnn_s = mre_s->expert_rnn_s[i];
        rnn_jacobian_matrix(mre_tmp_matrix[i], rnn_s->rnn_p, (T==0) ?
                rnn_s->init_c_state : rnn_s->c_state[T-1], rnn_s->c_state[T],
                rnn_s->out_state[T]);
    }

    I = 0; // index of output gate opening values
    for (int i = 0; i < expert_num; i++) {
        // index of input gate opening values
        J = expert_num * (gn_delay_length-1);
        for (int j = 0; j < expert_num; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i][j];
        }
        // index of context of the gating network
        J = dimension - gn_s->rnn_p->c_state_size;
        for (int j = 0; j < gn_s->rnn_p->c_state_size; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i][j + expert_num];
        }
    }

    I = expert_num * gn_delay_length; // index of context for each expert RNN
    for (int k = 0; k < expert_num; k++) {
        rnn_s = mre_s->expert_rnn_s[k];
        J = I; // index of context for each expert RNN
        for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
            for (int j = 0; j < rnn_s->rnn_p->c_state_size; j++) {
                matrix[I+i][J+j] =
                    mre_tmp_matrix[k][i + rnn_s->rnn_p->out_state_size][j];
            }
        }
        I += rnn_s->rnn_p->c_state_size;
    }
    // index of context of the gating network
    I = dimension - gn_s->rnn_p->c_state_size;
    for (int i = 0; i < gn_s->rnn_p->c_state_size; i++) {
        // index of input gate opening values
        J = expert_num * (gn_delay_length-1);
        for (int j = 0; j < expert_num; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i + expert_num][j];
        }
        J = I; // index of context of the gating network
        for (int j = 0; j < gn_s->rnn_p->c_state_size; j++) {
            matrix[I+i][J+j] = gn_tmp_matrix[i + expert_num][j + expert_num];
        }
    }

    for (int t = 1; t < gn_delay_length; t++) {
        // index of output gate opening values at time T-t
        I = expert_num * t;
        // index of output gate opening values at time T-t+1
        J = expert_num * (t-1);
        for (int i = 0; i < expert_num; i++) {
            matrix[I+i][J+i] = 1;
        }
    }

    return matrix;
}


double** mre_jacobian_for_lyapunov_spectrum (
        const double* vector,
        int n,
        int t,
        double** matrix,
        void *obj)
{
    struct mre_lyapunov_info *ml_info = (mre_lyapunov_info*)obj;
    const struct rnn_state *gn_s = ml_info->gn_s;
    const struct mre_state *mre_s = ml_info->mre_s;
    const int T = t + ml_info->truncate_length;

    assert(t >= 0);
    assert(gn_s->length > T);
    assert(mre_s->length > T);
    assert(vector == ml_info->state[t]);
    assert(n == ml_info->dimension);

    if (mre_s->mre->in_state_size == 0) {
        jacobian_matrix_without_input(matrix, ml_info->tmp_mre_matrix,
                ml_info->tmp_gn_matrix, ml_info->dimension, mre_s, gn_s,
                ml_info->gn_delay_length, T);
    } else {
        jacobian_matrix_with_delay(matrix, ml_info->tmp_mre_matrix,
                ml_info->tmp_gn_matrix, ml_info->dimension, mre_s, gn_s,
                ml_info->mre_delay_length, ml_info->gn_delay_length, T);
    }

    return matrix;
}


void reset_mre_lyapunov_info (struct mre_lyapunov_info *ml_info)
{
    const struct mre_state *mre_s = ml_info->mre_s;
    const struct rnn_state *gn_s = ml_info->gn_s;
    const int length = ml_info->length;
    const int mre_delay_length = ml_info->mre_delay_length;
    const int gn_delay_length = ml_info->gn_delay_length;
    const int truncate_length = ml_info->truncate_length;

    for (int n = 0; n < length; n++) {
        int I = 0;
        for (int k = mre_delay_length-1; k >= 0; k--) {
            const int N = n + truncate_length + k;
            if (N < mre_delay_length) {
                if (N < mre_s->length) {
                    for (int i = 0; i < mre_s->mre->in_state_size; i++, I++) {
                        ml_info->state[n][I] =
                            mre_s->expert_rnn_s[0]->in_state[N][i];
                    }
                } else {
                    for (int i = 0; i < mre_s->mre->in_state_size; i++, I++) {
                        ml_info->state[n][I] = 0;
                    }
                }
            } else {
                for (int i = 0; i < mre_s->mre->in_state_size; i++, I++) {
                    ml_info->state[n][I] =
                        mre_s->out_state[N-mre_delay_length][i];
                }
            }
        }
        for (int k = gn_delay_length-1; k >= 0; k--) {
            const int N = n + truncate_length + k;
            if (N < gn_delay_length) {
                if (N < gn_s->length) {
                    for (int i = 0; i < mre_s->mre->expert_num; i++, I++) {
                        ml_info->state[n][I] = gn_s->in_state[N][i];
                    }
                } else {
                    for (int i = 0; i < mre_s->mre->expert_num; i++, I++) {
                        ml_info->state[n][I] = 0;
                    }
                }
            } else {
                for (int i = 0; i < mre_s->mre->expert_num; i++, I++) {
                    ml_info->state[n][I] =
                        gn_s->out_state[N-gn_delay_length][i];
                }
            }
        }
        if (n + truncate_length == 0) {
            for (int k = 0; k < mre_s->mre->expert_num; k++) {
                const struct rnn_state *rnn_s = mre_s->expert_rnn_s[k];
                for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++, I++) {
                    ml_info->state[n][I] = rnn_s->init_c_inter_state[i];
                }
            }
            for (int i = 0; i < gn_s->rnn_p->c_state_size; i++, I++) {
                ml_info->state[n][I] = gn_s->init_c_inter_state[i];
            }
        } else {
            const int N = n + truncate_length;
            for (int k = 0; k < mre_s->mre->expert_num; k++) {
                const struct rnn_state *rnn_s = mre_s->expert_rnn_s[k];
                for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++, I++) {
                    ml_info->state[n][I] = rnn_s->c_inter_state[N-1][i];
                }
            }
            for (int i = 0; i < gn_s->rnn_p->c_state_size; i++, I++) {
                ml_info->state[n][I] = gn_s->c_inter_state[N-1][i];
            }
        }
    }
}


/* this function returns the Lyapunov spectrum of mixture of RNN experts */
double* mre_lyapunov_spectrum (
        struct mre_lyapunov_info *ml_info,
        double *spectrum,
        int spectrum_size)
{
    reset_mre_lyapunov_info(ml_info);
    return lyapunov_spectrum((const double* const*)ml_info->state,
            ml_info->length, spectrum_size, ml_info->dimension, 1,
            mre_jacobian_for_lyapunov_spectrum, ml_info, spectrum, NULL, NULL);
}

