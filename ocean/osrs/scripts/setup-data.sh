#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OSRS_ROOT="${ROOT}/ocean/osrs"
MANIFEST="${OSRS_ASSET_MANIFEST:-${OSRS_ROOT}/asset_manifest.json}"
DATA_DIR="${OSRS_DATA_DIR:-${OSRS_ROOT}/data}"
DOWNLOAD_DIR="${OSRS_ASSET_DOWNLOAD_DIR:-${DATA_DIR}/.download}"
MANIFEST_TOOL="${OSRS_ROOT}/scripts/osrs_asset_manifest.py"
ARCHIVE_TSV="${DOWNLOAD_DIR}/archive.tsv"
MISSING_TSV="${DOWNLOAD_DIR}/missing-required.tsv"

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "setup-osrs-data: missing required command: $1" >&2
        exit 1
    fi
}

verify_sha256() {
    local expected="$1"
    local path="$2"
    if command -v sha256sum >/dev/null 2>&1; then
        echo "${expected}  ${path}" | sha256sum -c -
        return
    fi
    if command -v shasum >/dev/null 2>&1; then
        echo "${expected}  ${path}" | shasum -a 256 -c -
        return
    fi
    echo "setup-osrs-data: missing required command: sha256sum or shasum" >&2
    exit 1
}

download() {
    local url="$1"
    local out="$2"
    echo "setup-osrs-data: downloading ${url}"
    curl -fL --retry 3 --retry-delay 2 -o "${out}" "${url}"
}

need curl
need python3
need tar

mkdir -p "${DOWNLOAD_DIR}" "${DATA_DIR}"

python3 "${MANIFEST_TOOL}" archive-tsv "${MANIFEST}" > "${ARCHIVE_TSV}"
IFS=$'\t' read -r archive_name archive_url archive_sha strip_components < "${ARCHIVE_TSV}"

if [ "${OSRS_ASSET_SETUP_FORCE:-0}" != "1" ]; then
    if python3 "${MANIFEST_TOOL}" missing-required "${MANIFEST}" "${DATA_DIR}" > "${MISSING_TSV}"; then
        echo "setup-osrs-data: loose assets satisfy ${archive_name}"
        exit 0
    fi
    missing_count=$(grep -c $'\t' "${MISSING_TSV}" || true)
    echo "setup-osrs-data: missing ${missing_count} required loose assets; installing ${archive_name}"
fi

archive_path="${DOWNLOAD_DIR}/${archive_name}"
tmp_archive="${DOWNLOAD_DIR}/.${archive_name}.tmp"

if [ ! -f "${archive_path}" ] || ! verify_sha256 "${archive_sha}" "${archive_path}" >/dev/null 2>&1; then
    rm -f "${tmp_archive}"
    download "${archive_url}" "${tmp_archive}"
    verify_sha256 "${archive_sha}" "${tmp_archive}"
    mv "${tmp_archive}" "${archive_path}"
fi

echo "setup-osrs-data: extracting ${archive_name} into ${DATA_DIR}"
tar_args=(-xzf "${archive_path}" --exclude='._*' --exclude='*/._*')
if [ "$(uname -s)" = "Linux" ]; then
    tar_args+=(--warning=no-unknown-keyword)
fi
tar_args+=(--strip-components="${strip_components}" -C "${DATA_DIR}")
tar "${tar_args[@]}"
find "${DATA_DIR}" -name '._*' -delete

if ! python3 "${MANIFEST_TOOL}" missing-required "${MANIFEST}" "${DATA_DIR}" > "${MISSING_TSV}"; then
    echo "setup-osrs-data: install did not satisfy manifest" >&2
    while IFS=$'\t' read -r group_name asset_path; do
        [ -n "${asset_path}" ] || continue
        echo "setup-osrs-data: missing ${group_name}: ${asset_path}" >&2
    done < "${MISSING_TSV}"
    exit 1
fi

echo "setup-osrs-data: ready"
