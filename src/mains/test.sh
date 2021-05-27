#!/usr/bin/env bash

CONFIG1=${1:-none}
CONFIG2=${2:-none}

if [ "$CONFIG1" = "none" ]; then
  python3 test_reliabilities.py --leonhard
else
  python3 test_reliabilities.py --leonhard --config $CONFIG1
fi

if [ "$CONFIG2" = "none" ]; then
    python3 test.py --leonhard
  else
    python3 test.py --leonhard --config $CONFIG2
fi
