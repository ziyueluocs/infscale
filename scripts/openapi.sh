#!/usr/bin/env bash

set -e

pwd=$PWD

if [[ $pwd = *'scripts'* ]]; then
    echo "Run this script in ${pwd/\/scripts/}"
fi

# note: make sure that controller is not running
# kill any running controller before running this script
# by with the following command: pkill -f "python -m infscale run controller"

python -m infscale run controller &
PID=$!
echo "wait for 10 seconds"
sleep 10

input_path=http://localhost:8080/openapi.json
openapi-generator generate \
                  -c ./scripts/generator-config.yaml \
                  -i ${input_path} -g python \
                  -o ./infscale/

# do other stuff
kill $PID
echo
