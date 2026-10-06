#!/bin/bash
# TurboTab: double-click to start it on a Mac.
#
# The first time, macOS may refuse a file downloaded from the internet: right-click this file,
# choose Open, then Open again. The first start sets TurboTab up (a few minutes, once); after that
# it takes seconds. Keep the window open while you work; closing it stops TurboTab.
exec bash "$(dirname "$0")/turbotab.sh" "$@"
