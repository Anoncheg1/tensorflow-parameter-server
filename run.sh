#!/bin/sh
#set +m # disable job control in order to allow lastpipe
#shopt -s lastpipe
set -u # Report Non-Existent Variables
# set -e # It terminates the execution when the error occurs. (does not work with piped commands. use Set -eo pipefail)
set -o pipefail # exit execution if one of the commands in the pipe fails.

who=$(python -c "import tensorflow as tf ; v=tf.distribute.cluster_resolver.TFConfigClusterResolver() ; print(v.task_type, v.task_id)" | tr " " "_")
rm ${who}.log
echo WHO=${who}
pgrep python | xargs kill -s 9
if [ $1 == "kill" ]; then return ; fi
script -c 'python smallmodel.py' |& sed -u "s/^/${who}: /" |& perl -ne 'use IO::Handle ; printf "%s %s",  scalar time(), $_ ; STDOUT->autoflush(1) ;' |& tee -a ${who}.log
