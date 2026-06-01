#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run-vanilla-reload-probe.sh \
    --server-root <root> \
    --jar <server-jar> \
    --setup-commands <setup.txt> \
    --observe-commands <observe.txt> \
    [--java-home <path>] \
    [--delay-ms 150] \
    [--ready-timeout 30] \
    [--log-prefix <prefix>]

Runs a two-phase vanilla probe against the same isolated server root:
1. start the server, execute the setup commands, then stop;
2. restart the server, execute the post-reload observation commands, then stop.
USAGE
}

server_root=""
jar_path=""
setup_commands=""
observe_commands=""
java_home="${JAVA_HOME:-}"
delay_ms=150
ready_timeout=30
log_prefix="reload-probe"

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
    --setup-commands)
      setup_commands=$2
      shift 2
      ;;
    --observe-commands)
      observe_commands=$2
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
    --log-prefix)
      log_prefix=$2
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

if [[ -z "$server_root" || -z "$jar_path" || -z "$setup_commands" || -z "$observe_commands" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -d "$server_root" ]]; then
  echo "Server root does not exist: $server_root" >&2
  exit 1
fi

if [[ ! -f "$setup_commands" ]]; then
  echo "Setup commands file does not exist: $setup_commands" >&2
  exit 1
fi

if [[ ! -f "$observe_commands" ]]; then
  echo "Observe commands file does not exist: $observe_commands" >&2
  exit 1
fi

setup_log="$server_root/${log_prefix}-setup.log"
observe_log="$server_root/${log_prefix}-observe.log"

common_args=(
  --server-root "$server_root"
  --jar "$jar_path"
  --delay-ms "$delay_ms"
  --ready-timeout "$ready_timeout"
)

if [[ -n "$java_home" ]]; then
  common_args+=(--java-home "$java_home")
fi

tools/run-vanilla-probe.sh \
  "${common_args[@]}" \
  --commands "$setup_commands" \
  --output "$setup_log"

tools/run-vanilla-probe.sh \
  "${common_args[@]}" \
  --commands "$observe_commands" \
  --output "$observe_log"

echo "Reload probe setup log: $setup_log"
echo "Reload probe observe log: $observe_log"
