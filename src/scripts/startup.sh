#!/usr/bin/env bash

curl curl -X POST -H 'Content-type: application/json' --data "{
    'attachments': [
        {
            'fallback': 'VM instance is running.',
            'color': '#FF8C00',
            'title': 'Instance Startup',
            'text': 'VM instance is running.',
            'footer': 'Beasty',
            'footer_icon': 'https://png2.kisspng.com/sh/6e14a20957f7cf13d844ea472ad8fce4/L0KzQYm3U8I5N5V4j5H0aYP2gLBuTfdwd5hxfZ9sbHB4dH73jPF1bpD3hZ9wb3BqfLa0gB9ueKZ5fZ9ubnfsfra0gBxwfZUye954dXSwc7F0kQV1cZ9sRadqYnO5RLO6UcI1a5Y3RqQEN0G8SIW3UcUzOmMATasDN0C6RXB3jvc=/kisspng-google-cloud-platform-google-compute-engine-cloud-cloud-computing-5abc64b3124ce2.297198401522295987075.png',
            'ts': 'date +\"%s\"'
        }
    ]
}" 'https://hooks.slack.com/services/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'