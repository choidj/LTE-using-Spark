#!/bin/bash

if [ "$SPARK_MODE" == "master" ]; then
    /opt/spark/sbin/start-master.sh -h 0.0.0.0
elif [ "$SPARK_MODE" == "worker" ]; then
    /opt/spark/sbin/start-worker.sh $SPARK_MASTER_URL
fi

tail -f /dev/null