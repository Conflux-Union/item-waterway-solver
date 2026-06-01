#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/run-vanilla-probe.sh \
    --server-root <root> \
    --jar <server-jar> \
    --commands <commands.txt> \
    [--java-home <path>] \
    [--delay-ms 150] \
    [--ready-timeout 30] \
    [--output <log-file>] \
    [--keep-server false]

Starts a vanilla server using a FIFO-backed stdin, waits until the server is
ready, sends one command per line from the commands file, and optionally stops
the server. Blank lines and lines beginning with '#' are ignored.
EOF
}

server_root=""
jar_path=""
commands_file=""
java_home="${JAVA_HOME:-}"
delay_ms=150
ready_timeout=30
output_log=""
keep_server=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-root)
      server_root=$2
      shift 2
      ;;
    --jar)
      jar_path=$2
      shift 2
      ;;
    --commands)
      commands_file=$2
      shift 2
      ;;
    --java-home)
      java_home=$2
      shift 2
      ;;
    --delay-ms)
      delay_ms=$2
      shift 2
      ;;
    --ready-timeout)
      ready_timeout=$2
      shift 2
      ;;
    --output)
      output_log=$2
      shift 2
      ;;
    --keep-server)
      keep_server=$2
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$server_root" || -z "$jar_path" || -z "$commands_file" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -d "$server_root" ]]; then
  echo "Server root does not exist: $server_root" >&2
  exit 1
fi

if [[ ! -f "$jar_path" ]]; then
  echo "Server jar does not exist: $jar_path" >&2
  exit 1
fi

if [[ ! -f "$commands_file" ]]; then
  echo "Commands file does not exist: $commands_file" >&2
  exit 1
fi

if [[ -z "$output_log" ]]; then
  timestamp=$(date +%Y%m%d-%H%M%S)
  output_log="$server_root/vanilla-probe-$timestamp.log"
fi

mkdir -p "$(dirname "$output_log")"

fifo="$server_root/server.stdin"
latest_log="$server_root/logs/latest.log"

if [[ -z "$java_home" ]]; then
  java_bin=java
else
  java_bin="$java_home/bin/java"
fi

rm -f "$fifo"
mkfifo "$fifo"
tail -f /dev/null > "$fifo" &
keeper_pid=$!
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    if [[ "$keep_server" != "true" ]]; then
      printf 'stop\n' > "$fifo" || true
      wait "$server_pid" || true
    fi
  fi
  if kill -0 "$keeper_pid" 2>/dev/null; then
    kill "$keeper_pid" || true
    wait "$keeper_pid" || true
  fi
  rm -f "$fifo"
}

trap cleanup EXIT

(
  cd "$server_root"
  exec "$java_bin" -Xms512M -Xmx4G -jar "$jar_path" nogui < "$fifo" > "$output_log" 2>&1
) &
server_pid=$!

ready_deadline=$((SECONDS + ready_timeout))
while true; do
  if grep -q 'Done (' "$output_log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Server exited before becoming ready." >&2
    tail -n 40 "$output_log" >&2 || true
    exit 1
  fi
  if (( SECONDS >= ready_deadline )); then
    echo "Timed out waiting for server readiness." >&2
    tail -n 40 "$output_log" >&2 || true
    exit 1
  fi
  sleep 1
done

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  [[ ${line:0:1} == "#" ]] && continue
  printf '%s\n' "$line" > "$fifo"
  python_delay=$(awk "BEGIN { printf \"%.3f\", $delay_ms / 1000 }")
  sleep "$python_delay"
done < "$commands_file"

if [[ "$keep_server" != "true" ]]; then
  printf 'stop\n' > "$fifo"
  wait "$server_pid"
  server_pid=""
fi

echo "Probe log: $output_log"
if [[ -f "$latest_log" ]]; then
  echo "Latest log: $latest_log"
fi
