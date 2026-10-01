"""Metadata-first access to public OpenNeuro snapshots.

Only the GraphQL file tree is queried during discovery. Annexed image bytes are
retrieved later, and only for paths explicitly selected in a cohort manifest.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from urllib.error import HTTPError

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size_bytes: int
    annexed: bool

    def to_dict(self) -> dict:
        return asdict(self)


class OpenNeuroClient:
    """Minimal standard-library client for OpenNeuro's documented GraphQL API."""

    def __init__(
        self, dataset_id: str, version: str | None = None, endpoint: str = OPENNEURO_GRAPHQL
    ):
        if version is None:
            raise ValueError(
                "A pinned OpenNeuro snapshot version is required for reproducible discovery."
            )
        self.dataset_id, self.version, self.endpoint = dataset_id, version, endpoint

    def _query(self, query: str, variables: dict) -> dict:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, payload, {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenNeuro HTTP {exc.code}: {detail}") from exc
        if body.get("errors"):
            raise RuntimeError(f"OpenNeuro GraphQL error: {body['errors']}")
        return body["data"]

    def files(self, tree: str | None = None, recursive: bool = False) -> list[dict]:
        """List one tree level. The tree identifier comes from a previous call."""
        query = """query($datasetId: ID!, $tag: String!, $tree: String, $recursive: Boolean) {
          snapshot(datasetId: $datasetId, tag: $tag) {
            files(tree: $tree, recursive: $recursive) { id filename size directory annexed }
          }
        }"""
        result = self._query(
            query,
            {
                "datasetId": self.dataset_id,
                "tag": self.version,
                "tree": tree,
                "recursive": recursive,
            },
        )
        snapshot = result.get("snapshot")
        if not snapshot:
            raise RuntimeError(f"No snapshot found for {self.dataset_id} version {self.version!r}.")
        return snapshot["files"]

    def subject_files(self, subject_tree: str, subject_name: str) -> list[RemoteFile]:
        """Recursively walk exactly one participant subtree, retaining file sizes."""
        out: list[RemoteFile] = []

        # The documented recursive API may return path-qualified names. Use it
        # when it does, avoiding thousands of per-directory requests.
        recursive_entries = self.files(subject_tree, recursive=True)
        if any("/" in entry["filename"] for entry in recursive_entries):
            for entry in recursive_entries:
                if not entry["directory"]:
                    name = entry["filename"]
                    path = name if name.startswith(f"{subject_name}/") else f"{subject_name}/{name}"
                    out.append(
                        RemoteFile(path, int(entry.get("size") or 0), bool(entry.get("annexed")))
                    )
            return out

        def walk(tree: str, prefix: PurePosixPath) -> None:
            for entry in self.files(tree):
                path = prefix / entry["filename"]
                if entry["directory"]:
                    walk(entry["id"], path)
                else:
                    out.append(
                        RemoteFile(
                            str(path), int(entry.get("size") or 0), bool(entry.get("annexed"))
                        )
                    )

        walk(subject_tree, PurePosixPath(subject_name))
        return out


def is_imaging_file(path: str) -> bool:
    return path.endswith(".nii") or path.endswith(".nii.gz")
