#!/bin/bash

MY_PROGRAM="bash" # For example, "apache2" or "nginx"
MY_USER="Play"
BUCKET_NAME="mt-buffer" # For example, "my-checkpoint-files" (without gs://)

echo "Shutting down!  Seeing if ${MY_PROGRAM} is running."

# Find the newest copy of $MY_PROGRAM
PIDS="$(pgrep -n ${MY_PROGRAM})"

if [[ ${?} -ne 0 ]]; then
  echo "${MY_PROGRAM} not running, shutting down immediately."
  exit 0
fi

echo "${MY_PROGRAM} is running: ${PIDS}."
kill -2 ${PIDS}

# Portable waitpid equivalent
while kill -0 ${PIDS}; do
   sleep 1
done

sleep 10

PIDS="$(pgrep ${MY_PROGRAM})"

if [[ ${?} -ne 0 ]]; then
  echo "${MY_PROGRAM} not running, shutting down immediately."
  exit 0
fi

echo "${MY_PROGRAM} is running: ${PIDS}."
kill -2 ${PIDS}

# Portable waitpid equivalent
while kill -0 ${PIDS}; do
   sleep 1
done

echo "All processed gracefully stopped."
exit 0