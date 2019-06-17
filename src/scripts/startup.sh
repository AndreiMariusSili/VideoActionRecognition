#!/usr/bin/env bash

curl curl -X POST -H 'Content-type: application/json' --data "{
    'attachments': [
        {
            'fallback': 'VM instance is running.',
            'color': '#FF8C00',
            'title': 'Instance Startup',
            'text': 'VM instance is running.',
            'footer': 'Beasty',
            'ts': 'date +\"%s\"'
        }
    ]
}" 'https://hooks.slack.com/services/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'