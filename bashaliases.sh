#!/usr/bin/env bash

alias log-create-one="cat chief_0.log ps_0.log worker_0.log worker_1.log | sed 's/\r//' | sort > log-$(date -I).log"

alias log-create-one-before="cat chief_0.log ps_0.log worker_0.log worker_1.log | sed 's/\r//' | sort | grep -B 9999999999 \"Step: ${1}\$\""

alias log-create-one-after="cat chief_0.log ps_0.log worker_0.log worker_1.log | sed 's/\r//' | sort | grep -A 9999999999 \"Step: ${1}$\""

_read-log(){
    if [ -z "$1" ]; then
        echo cat chief_?.log ps_?.log worker_?.log
        log=$(cat chief_?.log ps_?.log worker_?.log) # Carefull!
    elif [ -f "$1" ]; then
        echo cat $1
        log=$(cat $1) # Carefull!
    else
        echo cat "$1"/chief_?.log "$1"/ps_?.log "$1"/worker_?.log
        log=$(cat "$1"/chief_?.log "$1"/ps_?.log "$1"/worker_?.log) # Carefull!
    fi
}

log-full-report() {
    _read-log $@ # get $log
    export TZ="Europe/Moscow" # set timezone
    steps=$(echo "$log" | grep "Step: [0-9]" | sort -ns -k1)
    chief=$(echo "$log" | grep "chief_.:")
    echo

    # -- Batch size
    echo Batch size: $(echo "$chief" | grep -o "batch_size: [0-9]*" | grep -o "[0-9]*")
    echo

    # -- Parameters count
    echo Parameters count: $(echo "$chief" | grep -io "Total params.*" | tail -n 1)
    echo

    # -- Input layout of model:
    echo Input layout of model: $(echo "$chief" | grep -io "input_1.*" | tail -n 1)
    echo

    # -- Output layout of model:
    echo Output layout of model: $(echo "$chief" | grep -io "d-out (Dense).*" | tail -n 1)
    echo


    # -- Dataset size:
    echo Dataset size: $(echo "$chief" | grep -io "df.shape.*" | tail -n 1)
    echo



    # -- time - first and last step in all epochs
    sec_b=$(echo "$steps" | head -n 1 | cut -d ' ' -f 1)
    sec_e=$(echo "$steps" | tail -n 1 | cut -d ' ' -f 1)
    sec_count=$(( ${sec_e} - ${sec_b} ))
    echo -e Time of start: "\t\t"$(date -d @${sec_b})
    echo -e Time of finish: "\t"$(date -d @${sec_e})
    echo -e Duration: "\t"$(($sec_count / 60 )) minutes, "\t"$(($sec_count / 60 /60 )) hours
    echo

    # -- steps - grep "steps in epoch" + last record with "Step:"
    steps_total=$(echo "$chief" | grep -io "steps in epoch: [0-9]*" | cut -d ' ' -f 4)
    echo "Total steps in epoch:" $steps_total
    echo
    steps_processed=$(echo "$steps" | tail -n 1 | cut -d ' ' -f 4)
    echo Total steps processed: $steps_processed
    echo
    # -- total rate - total steps / seconds in first and last step in all epochs
    rate=$(python -c "print(round( $steps_processed / $sec_count , 4))")
    echo "Total rate step per second:" $rate
    echo

    # -- Predict time: - require total_steps
    # echo steps_total=$steps_total
    # echo steps_processed=$steps_processed
    # echo sec_count=$sec_count
    # # echo steps=$steps
    # echo "round( ($steps_total - $steps_processed) * ( $sec_count / $steps_processed)"
    # echo $steps_total - $steps_processed
    steps_left=$(($steps_total - $steps_processed))
    left_seconds=$(python -c "print(round( $steps_left * ( $sec_count / $steps_processed) , 0))")
    left_minutes=$(python -c "print(round( ($steps_left * ( $sec_count / $steps_processed))/60 , 2))")
    left_hours=$(python -c "print(round( ($steps_left * ( $sec_count / $steps_processed))/60/60 , 2))")
    left_days=$(python -c "print(round( ($steps_left * ( $sec_count / $steps_processed))/60/60/60 , 2))")
    echo Predict time left until finish: seconds: $left_seconds, minutes: $left_minutes, hours: $left_hours, days: $left_days
    echo



    # -- worker's calc statistic --
    # - count workers
    workers_count=$(echo "$log" | grep -o "worker_.:" | sort | uniq | wc -l)
    echo workers: $workers_count
    echo
    echo "Steps in second:"
    for i in $(seq 0 $(($workers_count - 1))) ; do
        steps_w=$(echo "$steps" | grep "worker_${i}:")
        steps_count=$(echo "$steps_w" | wc -l)
        sec=$(echo "$steps_w" | cut -d ' ' -f 1)
        sec_b=$(echo "$sec" | head -n 1)
        sec_e=$(echo "$sec" | tail -n 1)
        sec_count=$(( ${sec_e} - ${sec_b} ))
        rate=$(python -c "print(round( $steps_count / $sec_count , 4))")
        echo worker_${i}: ${rate}, duration: $((${sec_count}/60))m, $steps_count steps, start: $(date +%X -d @${sec_b}), finish: $(date +%X -d @${sec_e})
	echo "$steps_w" | grep -o "Step: [0-9]*" | grep -o "[0-9]*" > /tmp/worker_$i
    done
    echo
    echo Count of same steps for Worker_0 and Worker_1: $(python -c 'a1=set([int(x) for x in open("/tmp/worker_1").readlines()]); a0=set([int(x) for x in open("/tmp/worker_0").readlines()]) ; print(len(a1.intersection(a0)))')
    # --

}

log-watch() { # deps: no
    if [ -z "$1" ]; then
        watch "cat chief_?.log ps_?.log worker_?.log | sort -ns -k1 | tail -n 23"
    else
        watch "cat $1 | tail "
    fi
}

log-rateofsteps-between() { # deps: _read-log
    _read-log $@ # get $log
    b=$1
    e=$2
    sec_b=$(echo "$log" | sed 's/\r//' | sort -ns -k1 | grep -A 9999999 "Step: 100$" | grep "Step: ${b}$" | cut -d ' ' -f 1)
    sec_e=$(cat chief_0.log ps_0.log worker_0.log worker_1.log | sed 's/\r//' | sort -ns -k1 | grep -A 9999999 "Step: 100$" | grep "Step: ${e}$" | cut -d ' ' -f 1)
    sec_count=$((${sec_e} - ${sec_b}))
    steps_count=$(($e - $b))
    echo count: $count, begin second: $sec_b, end second: $sec_e, cound seconds: $sec_count
    python -c "print(round( $steps_count / $sec_count , 4))"
}


log-workers-balance() { # deps: no
    w0=$(cat worker_0.log | grep Step: | grep worker_0 | wc -l)
    w1=$(cat worker_1.log | grep Step: | grep worker_0 | wc -l)
    echo "worker_0:${w0}" $(python -c "print(round($w0 / ( $w0+$w1 ) ,4))")
    echo "worker_1:${w1}" $(python -c "print(round($w1 / ( $w0+$w1 ) ,4))")
}

log-dataset-stats() {
    s0=$(cat worker_0.log  | grep Dataset.map | head -n 1 | cut -d ' ' -f 1)
    d0=$(cat worker_0.log | grep $s0 | grep -o 'batch:.*' | sort | uniq)
    s1=$(cat worker_1.log  | grep Dataset.map | head -n 1 | cut -d ' ' -f 1)
    d1=$(cat worker_1.log | grep $s1 | grep -o 'batch:.*' | sort | uniq)
    echo first len: w0: $(echo -e "$d0" | wc -l) w1: $(echo -e "$d1" | wc -l)
    # echo -e "$d0" > /tmp/t0
    # echo -e "$d1" > /tmp/t1
    # echo first same: $(python -c "print(len(set(open('/tmp/t0').readlines()).intersection(set(open('/tmp/t1').readlines()))))")
    echo first same $(cat <(echo -e "$d0")  <(echo -e "$d1") | sort | uniq -c | grep -o '[0-9].*' |cut -d ' ' -f 1 | grep '[2-9]' | wc -l)
    # echo first diff: $(diff <(echo -e "$d0")  <(echo -e "$d1") | wc -l)
    echo
    echo last step: $(cat worker_?.log | grep "Step: [0-9]" | sort -ns -k1 | tail -n 1 | cut -d ' ' -f 4)
    echo
    d0=$(cat worker_0.log | grep -v $s0 | grep "Step: [0-9]" | grep -o 'batch:.*' | sort | uniq)
    d1=$(cat worker_1.log | grep -v $s1 | grep "Step: [0-9]" | grep -o 'batch:.*' | sort | uniq)
    echo second len: w0: $(echo -e "$d0" | wc -l) w1: $(echo -e "$d1" | wc -l)
    echo second diff: $(diff <(echo -e "$d0")  <(echo -e "$d1") | wc -l)
    # echo -e "$d0" > /tmp/t0
    # echo -e "$d1" > /tmp/t1
    # echo second same: $(python -c "print(len(set(open('/tmp/t0').readlines()).intersection(set(open('/tmp/t1').readlines()))))")
    echo second same $(cat <(echo -e "$d0")  <(echo -e "$d1") | sort | uniq -c | grep -o '[0-9].*' |cut -d ' ' -f 1 | grep '[2-9]' | wc -l)

}
