#!/usr/bin/env bash

CONFIG1=${1:-none}
CONFIG2=${2:-none}

if [ "$CONFIG1" = "none" ]; then
  python3 test_reliabilities_old.py --leonhard
else
  python3 test_reliabilities_old.py --leonhard --config $CONFIG1
fi

if [ "$CONFIG2" = "none" ]; then
    python3 test_old.py --leonhard
  else
    python3 test_old.py --leonhard --config $CONFIG2
fi
