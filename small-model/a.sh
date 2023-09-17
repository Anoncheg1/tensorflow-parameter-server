#!/bin/sh
# apt install -y moreutils
#set +m # disable job control in order to allow lastpipe
#shopt -s lastpipe
# set -u # Report Non-Existent Variables
# set -e # It terminates the execution when the error occurs. (does not work with piped commands. use Set -eo pipefail)
set -o pipefail # exit execution if one of the commands in the pipe fails.

who=$(python -c "import tensorflow as tf ; v=tf.distribute.cluster_resolver.TFConfigClusterResolver() ; print(v.task_type, v.task_id)" | tr " " "_")
rm ${who}.log
echo WHO=${who}
pgrep python | xargs kill -s 9
if [ $1 == "kill" ]; then return ; fi
#exec > >(tee -a ${who}.log)
#exec 2>&1
# script -c 'python param.py' 2>&1 | sed "s/^/${who}: /" | ts | tee -a ${who}.log
script -c 'python smallmodel.py' |& sed -u "s/^/${who}: /" |& perl -ne 'use IO::Handle ; printf "%s %s",  scalar time(), $_ ; STDOUT->autoflush(1) ;' |& tee -a ${who}.log

# |& perl -ne 'use IO::Handle ; printf "%s %s",  scalar time(), $_ ; STDOUT->autoflush(1) ;' > a.log

# |& sed "s/^/${who}: /" |& perl -ne 'use IO::Handle ; printf "%s %s",  scalar time(), $_ ; STDOUT->autoflush(1) ; ' &>> a.log
# python param.py |&cat # 2>&1 | sed "s/^/${who}: /"
# echo wtf 2>&1 | cat | perl -ne 'printf "%s %s",  scalar time(), $_'
# echo asd    | sed "s/^/${who}: /" 2>&1 | perl -ne 'printf "%s %s",  scalar time(), $_' | cat
