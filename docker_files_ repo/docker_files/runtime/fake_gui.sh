#!/bin/bash
#
# IMPORTANT: Change this file only in directory Standalone!
# https://stackoverflow.com/questions/46613117/why-does-conda-install-tk-not-work-in-my-docker-container-even-though-it-says-it/46613321?noredirect=1#comment80181642_46613321

#echo 'running fake_gui.sh'

Xvfb :1 -screen 0 1024x768x16 &> xvfb.log  &

source /opt/bin/functions.sh

export GEOMETRY="$SCREEN_WIDTH""x""$SCREEN_HEIGHT""x""$SCREEN_DEPTH"

function shutdown {
  kill -s SIGTERM $NODE_PID
  wait $NODE_PID
}

if [ ! -z "$SE_OPTS" ]; then
  echo "appending selenium options: ${SE_OPTS}"
fi

SERVERNUM=$(get_server_num)

rm -f /tmp/.X*lock

xvfb-run -n $SERVERNUM --server-args="-screen 0 $GEOMETRY -ac +extension RANDR" \
  java ${JAVA_OPTS} -jar /opt/selenium/selenium-server-standalone.jar \
  ${SE_OPTS} &
NODE_PID=$!

trap shutdown SIGTERM SIGINT
wait $NODE_PID
