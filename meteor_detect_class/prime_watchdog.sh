# PLS ACTIVE VIRTUAL ENVIRONMENT BEFORE RUNNING THIS SCRIPT

while true; do

    current_time=$(date "+%Y.%m.%d-%H.%M.%S")

    echo
    echo "RESTART PRIME DETECTION at $current_time"

    echo "\n" >> log.txt
    echo "RESTART PRIME DETECTION at $current_time" >> log.txt
    python prime_detection.py >> log.txt


    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    echo "CRASHED... at $current_time"
    sleep 3

done