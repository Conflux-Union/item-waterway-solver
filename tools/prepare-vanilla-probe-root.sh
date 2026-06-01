#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/prepare-vanilla-probe-root.sh <base-root> <dest-root> <world-name> <port>

Creates an isolated vanilla server root by copying an existing root and
rewriting `level-name` and `server-port` in `server.properties`.

The destination directory must not already exist.
EOF
}

if [[ $# -ne 4 ]]; then
  usage >&2
  exit 1
fi

base_root=$1
dest_root=$2
world_name=$3
port=$4

if [[ ! -d "$base_root" ]]; then
  echo "Base root does not exist: $base_root" >&2
  exit 1
fi

if [[ -e "$dest_root" ]]; then
  echo "Destination already exists: $dest_root" >&2
  exit 1
fi

if [[ ! "$port" =~ ^[0-9]+$ ]]; then
  echo "Port must be an integer: $port" >&2
  exit 1
fi

mkdir -p "$dest_root"
cp -a "$base_root/." "$dest_root/"

properties_file="$dest_root/server.properties"
if [[ ! -f "$properties_file" ]]; then
  echo "Missing server.properties in destination root: $properties_file" >&2
  exit 1
fi

perl -0pi -e "s/^level-name=.*/level-name=$world_name/m; s/^server-port=.*/server-port=$port/m" "$properties_file"

echo "Prepared isolated probe root:"
echo "  root: $dest_root"
echo "  world: $world_name"
echo "  port: $port"
