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

#ifndef MREGN_H
#define MREGN_H


#include "rnn.h"
#include "mre.h"

void mregn_forward_dynamics (
        struct mre_state *mre_s,
        struct rnn_state *gn_s,
        int gn_delay_length);

void mregn_forward_dynamics_forall (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int gn_delay_length);

void mregn_forward_dynamics_in_closed_loop (
        struct mre_state *mre_s,
        struct rnn_state *gn_s,
        int mre_delay_length,
        int gn_delay_length);

void mregn_forward_dynamics_in_closed_loop_forall (
        struct mixture_of_rnn_experts *mre,
        struct recurrent_neural_network *gn,
        int mre_delay_length,
        int gn_delay_length);

#endif

