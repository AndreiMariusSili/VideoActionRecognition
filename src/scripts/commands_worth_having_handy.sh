#!/usr/bin/env bash

gcloud compute instances add-metadata beasty-nl --metadata-from-file shutdown-script=shutdown.sh --project=andreis-playground --zone=europe-west4-a
gcloud compute instances add-metadata beasty-nl --metadata-from-file startup-script=startup.sh --project=andreis-playground --zone=europe-west4-a