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

#ifndef MAIN_H
#define MAIN_H


/* parameters for mixture-of-RNN-experts learning */
typedef struct main_parameters {
    unsigned long seed;                 // random seed
    long epoch_size;                    // number of epochs in learning
    /* if use_adaptive_lr!=0, learning rate is adaptively changed */
    int use_adaptive_lr;
    double rho;                         // learning rate
    double momentum;                    // momentum of learning
    int expert_num;                     // number of experts
    int c_state_size;                   // number of context neurons
    int delay_length;                   // feedback delay

    /*
     * If fixed_[weight|threshold|tau|init_c_state|sigma|gate] != 0,
     * then it does not change by learning.
     */
    int fixed_weight;
    int fixed_threshold;
    int fixed_tau;
    int fixed_init_c_state;
    int fixed_sigma;
    int fixed_gate;

    /* connection between input neurons to context neurons */
    char *connection_i2c;
    /* connection between context neurons to context neurons */
    char *connection_c2c;
    /* connection between context neurons to output neurons */
    char *connection_c2o;

    char *const_init_c;                 // constant initial context values
    char *init_tau;                     // initial value(s) of time constant
    double init_sigma;                  // initial value of sigma
    /* control parameter for learning of gate opening values */
    double gamma;
    /*
     * type of prior distribution for gate opening values
     * (0:NO_DISTRIBUTION, 1:GAUSS_DISTRIBUTION, 2:CAUCHY_DISTRIBUTION)
     */
    int gate_prior_distribution;
    double prior_strength;              // strength of prior distribution

    /* damping rate (=lambda) and scaling rate (=alpha) for prior_strength */
    double lambda;
    double alpha;
} main_parameters;


/* parameters of File IO */
typedef struct io_parameters {
    /* file names */
    char *state_filename;
    char *closed_state_filename;
    char *gate_filename;
    char *weight_filename;
    char *threshold_filename;
    char *tau_filename;
    char *sigma_filename;
    char *init_filename;
    char *adapt_lr_filename;
    char *error_filename;
    char *closed_error_filename;

    char *save_filename;
    char *load_filename;

    /* interval for printing data */
    struct print_interval {
        long interval;
        long init;
        long end;
        int use_logscale_interval;
        int _set_interval_flag;
        int _set_init_flag;
        int _set_end_flag;
        int _set_use_logscale_interval_flag;
    } default_interval;
    struct print_interval interval_for_state_file;
    struct print_interval interval_for_closed_state_file;
    struct print_interval interval_for_gate_file;
    struct print_interval interval_for_weight_file;
    struct print_interval interval_for_threshold_file;
    struct print_interval interval_for_tau_file;
    struct print_interval interval_for_sigma_file;
    struct print_interval interval_for_init_file;
    struct print_interval interval_for_adapt_lr_file;
    struct print_interval interval_for_error_file;
    struct print_interval interval_for_closed_error_file;

    /* if verbose!=0, explain what is being done */
    int verbose;
} io_parameters;


typedef struct internal_parameters {
    double adapt_lr;                    // adaptive learning rate
    long init_epoch;                    // number of initial epochs
    int **has_connection_ci;
    int **has_connection_cc;
    int **has_connection_oc;
    double **connectivity_ci;
    double **connectivity_cc;
    double **connectivity_oc;
    int *const_init_c;
    double *init_tau;
} internal_parameters;


typedef struct general_parameters {
    struct main_parameters mp;
    struct io_parameters iop;
    struct internal_parameters inp;
} general_parameters;

#endif

