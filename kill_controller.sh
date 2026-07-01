#!/bin/bash
echo "Looking for ghost osken-manager processes..."

# The [o] trick prevents grep from matching itself
PIDS=$(ps aux | grep "[o]sken-manager" | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No osken-manager processes found running."
else
    echo "Found process IDs: $PIDS"
    echo "Force killing them now..."
    kill -9 $PIDS
    echo "Done! The old controller has been successfully terminated."
fi
