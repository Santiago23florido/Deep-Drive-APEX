#! /usr/bin/env bash

# This script copies the necessary files to the Raspberry Pi for running the autonomous car software.
# It assumes that you have SSH access to the Raspberry Pi.

# Define the Raspberry Pi's IP address and username
PI_USER="ensta"
PI_PSWD="ensta"
PI_HOST="10.42.0.96"

# List of files and directories to copy
FOLDERS_TO_COPY=(
    "code/"
    "code_test/"
    "scripts/"
    "serial/"
    "scripts/"   
)

# Get full root path of repository
LOCAL_FULL_PATH="$(dirname "$(realpath "$0")")"
DEST_DIR="/home/ensta/Voiture-Autonome/"

# Function to set the Raspberry Pi host
set_pi_host() {
    if [ -n "$1" ]; then
        PI_HOST="$1"
        echo "Inserted host value as $PI_HOST"
    else
        echo "Using previous host value: $PI_HOST"
    fi
}

# Obtain HOST as input (uncomment the next line to enable input)
read -p "Enter Raspberry Pi host IP (default: $PI_HOST): " user_input
set_pi_host "${user_input:-$PI_HOST}"


# Copy the files and directories to the Raspberry Pi
for FOLDER in "${FOLDERS_TO_COPY[@]}"; do
    echo "Copying $FOLDER from $LOCAL_FULL_PATH/$FOLDER to $PI_USER@$PI_HOST:$DEST_DIR"
    sshpass -p "$PI_PSWD" scp -r "$LOCAL_FULL_PATH/$FOLDER" "$PI_USER@$PI_HOST:$DEST_DIR$FOLDER"
done
