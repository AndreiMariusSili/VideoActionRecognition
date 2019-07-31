#!/bin/bash

MY_PROGRAM="bash" # For example, "apache2" or "nginx"

echo "Shutting down!  Seeing if ${MY_PROGRAM} is running."

curl curl -X POST -H 'Content-type: application/json' --data "{
    'attachments': [
        {
            'fallback': 'VM instance is stopped.',
            'color': '#FF8C00',
            'title': 'Instance Shutdown',
            'text': 'VM instance is stopped.',
            'footer': 'Beasty',
            'ts': 'date +\"%s\"'
        }
    ]
}" 'https://hooks.slack.com/services/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'

# Find the newest copy of $MY_PROGRAM
PIDS="$(pgrep -n ${MY_PROGRAM})"

if [[ "${?}" -ne 0 ]]; then
  echo "${MY_PROGRAM} not running, shutting down immediately."
  exit 0
fi

echo "${MY_PROGRAM} is running: ${PIDS}."
kill -2 "${PIDS}"

# Portable waitpid equivalent
while kill -0 "${PIDS}"; do
   sleep 1
done

sleep 10

PIDS="$(pgrep ${MY_PROGRAM})"

if [[ ${?} -ne 0 ]]; then
  echo "${MY_PROGRAM} not running, shutting down immediately."
  exit 0
fi

echo "${MY_PROGRAM} is running: ${PIDS}."
kill -2 "${PIDS}"

# Portable waitpid equivalent
while kill -0 "${PIDS}"; do
   sleep 1
done

echo "All processes gracefully stopped."
exit 0