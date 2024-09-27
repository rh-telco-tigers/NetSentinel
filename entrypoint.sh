#!/bin/bash
set -e

if [ "$#" -eq 0 ]; then
    python help.py
else
    exec "$@"
fi
