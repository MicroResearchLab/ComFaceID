#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

download_file() {
  local record_id="$1"
  local file_key="$2"
  local output_path="$3"
  local url="https://zenodo.org/api/records/${record_id}/files/${file_key}/content"

  mkdir -p "$(dirname "$output_path")"

  if [[ -f "$output_path" ]]; then
    echo "[skip] $output_path already exists"
    return 0
  fi

  echo "[download] ${file_key} -> ${output_path}"
  curl -fL --retry 3 --retry-delay 5 --continue-at - --output "$output_path" "$url"
}

main() {
  command -v curl >/dev/null 2>&1 || {
    echo "Error: curl is required but not installed." >&2
    exit 1
  }

  download_file 16676832 ".index" "$ROOT_DIR/base_model/.index"
  download_file 16676832 "checkpoint" "$ROOT_DIR/base_model/checkpoint"
  download_file 16676832 ".data-00000-of-00001" "$ROOT_DIR/base_model/.data-00000-of-00001"

  download_file 16739187 "final.pth" "$ROOT_DIR/class_model/class/final.pth"
  download_file 16739195 "final.pth" "$ROOT_DIR/class_model/superclass/final.pth"

  download_file 16682503 "final.pth" "$ROOT_DIR/fpr_model/final.pth"
  download_file 16679974 "CONAPUS_pubchem_export.csv" "$ROOT_DIR/fpr_database/CONAPUS_pubchem_export.csv"

  echo "All requested resources have been processed."
}

main "$@"