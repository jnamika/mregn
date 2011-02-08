#!/bin/sh
if test ! -d m4; then
    mkdir m4
fi
if test ! -d rnn/m4; then
    mkdir rnn/m4
fi
autoreconf -i
