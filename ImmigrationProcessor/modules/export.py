"""Export of processed documents as ZIP archives.

Two flavours are supported:

* ``Smith_John.zip`` - every document of a single client, flat inside the
  archive, and
* ``AllClients_YYYYMMDD_HHMM.zip`` - all clients, each in its own folder.

Archives are written to ``data/temp/exports`` and also returned as bytes so
Streamlit can offer them through ``st.download_button``.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from config import SETTINGS, Settings, get_logger
from modules.folder_manager import ClientFolder, FolderManager

logger = get_logger(__name__)


class ExportError(RuntimeError):
    """Raised when an archive cannot be created."""


@dataclass(frozen=True)
class ExportResult:
    """A created ZIP archive."""

    path: Path
    file_count: int
    clients: tuple[str, ...]

    @property
    def name(self) -> str:
        """File name of the archive."""
        return self.path.name

    @property
    def size_kb(self) -> float:
        """Archive size in kilobytes."""
        try:
            return self.path.stat().st_size / 1024
        except OSError:  # pragma: no cover - archive removed meanwhile
            return 0.0

    def read_bytes(self) -> bytes:
        """Return the archive content (for the download button)."""
        return self.path.read_bytes()


class Exporter:
    """Creates ZIP archives from the client folders."""

    def __init__(self, settings: Settings = SETTINGS, folder_manager: FolderManager | None = None) -> None:
        self._settings = settings
        self._folders = folder_manager or FolderManager(settings)

    @property
    def export_dir(self) -> Path:
        """Directory the archives are written to."""
        directory = self._settings.paths.temp / "exports"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # -- public API -------------------------------------------------------- #
    def export_client(self, client_key: str) -> ExportResult:
        """Zip every document of one client into ``<client_key>.zip``.

        Raises:
            ExportError: When the client folder is unknown or empty.
        """
        folder = self._folders.client_folder(client_key, create=False)
        if not folder.is_dir():
            raise ExportError(f"Unknown client folder: {client_key}")
        files = sorted(path for path in folder.glob("*.pdf") if path.is_file())
        if not files:
            raise ExportError(f"No documents to export for {client_key}")

        target = self.export_dir / f"{client_key}.zip"
        self._write_archive(target, ((path, path.name) for path in files))
        logger.info("Export created: %s (%d document(s))", target.name, len(files))
        return ExportResult(path=target, file_count=len(files), clients=(client_key,))

    def export_all(self) -> ExportResult:
        """Zip every client folder, each into its own directory inside the archive.

        Raises:
            ExportError: When no processed document exists at all.
        """
        clients: Sequence[ClientFolder] = [client for client in self._folders.list_clients() if client.files]
        if not clients:
            raise ExportError("There are no processed documents to export")

        target = self.export_dir / f"AllClients_{datetime.now():%Y%m%d_%H%M}.zip"
        entries: list[tuple[Path, str]] = [
            (path, f"{client.key}/{path.name}") for client in clients for path in client.files
        ]
        self._write_archive(target, entries)
        logger.info(
            "Export created: %s (%d client(s), %d document(s))", target.name, len(clients), len(entries)
        )
        return ExportResult(
            path=target,
            file_count=len(entries),
            clients=tuple(client.key for client in clients),
        )

    def export_paths(self, files: Sequence[Path], archive_name: str) -> ExportResult:
        """Zip an explicit list of files (used for ad-hoc downloads)."""
        existing = [Path(path) for path in files if Path(path).is_file()]
        if not existing:
            raise ExportError("None of the requested files exist")
        target = self.export_dir / (archive_name if archive_name.endswith(".zip") else f"{archive_name}.zip")
        self._write_archive(target, ((path, path.name) for path in existing))
        return ExportResult(path=target, file_count=len(existing), clients=())

    # -- internals --------------------------------------------------------- #
    def _write_archive(self, target: Path, entries: Iterable[tuple[Path, str]]) -> None:
        """Write *entries* (``(source, name_in_zip)``) into the archive *target*."""
        try:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for source, arcname in entries:
                    archive.write(source, arcname=arcname)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ExportError(f"Cannot write archive {target.name}: {exc}") from exc
