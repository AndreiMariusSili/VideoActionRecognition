#!/bin/bash

MY_PROGRAM="bash" # For example, "apache2" or "nginx"
MY_USER="Play"
BUCKET_NAME="mt-buffer" # For example, "my-checkpoint-files" (without gs://)

echo "Shutting down!  Seeing if ${MY_PROGRAM} is running."

curl curl -X POST -H 'Content-type: application/json' --data "{
    'attachments': [
        {
            'fallback': 'VM instance is stopped.',
            'color': '#FF8C00',
            'title': 'Instance Shutdown',
            'text': 'VM instance is stopped.',
            'footer': 'Beasty',
            'footer_icon': 'https://png2.kisspng.com/sh/6e14a20957f7cf13d844ea472ad8fce4/L0KzQYm3U8I5N5V4j5H0aYP2gLBuTfdwd5hxfZ9sbHB4dH73jPF1bpD3hZ9wb3BqfLa0gB9ueKZ5fZ9ubnfsfra0gBxwfZUye954dXSwc7F0kQV1cZ9sRadqYnO5RLO6UcI1a5Y3RqQEN0G8SIW3UcUzOmMATasDN0C6RXB3jvc=/kisspng-google-cloud-platform-google-compute-engine-cloud-cloud-computing-5abc64b3124ce2.297198401522295987075.png',
            'ts': 'date +\"%s\"'
        }
    ]
}" 'https://hooks.slack.com/services/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'

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

echo "All processes gracefully stopped."
exit 0