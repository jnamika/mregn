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

#ifndef PARAMETER_H
#define PARAMETER_H


#define EPOCH_SIZE 1000
#define PRINT_INTERVAL 100

#define RHO 0.001
#define MOMENTUM 0.9

#define EXPERT_NUM 2
#define C_STATE_SIZE 10

#define DELAY_LENGTH 1

#define INIT_TAU 5
#define INIT_SIGMA 0.0
#define GAMMA 1
#define GATE_PRIOR_DISTRIBUTION 1
#define PRIOR_STRENGTH 0
#define LAMBDA 1.0
#define ALPHA 0.0

#define STATE_FILENAME ""
#define CLOSED_STATE_FILENAME ""
#define GATE_FILENAME ""
#define WEIGHT_FILENAME ""
#define THRESHOLD_FILENAME ""
#define TAU_FILENAME ""
#define SIGMA_FILENAME ""
#define INIT_FILENAME ""
#define ADAPT_LR_FILENAME ""
#define ERROR_FILENAME "error.log"
#define CLOSED_ERROR_FILENAME ""
#define SAVE_FILENAME "mre.dat"
#define LOAD_FILENAME ""


#endif

