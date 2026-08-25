"""Ownership of the file system: uploads, temp work areas, client folders, review.

Layout managed by this module::

    data/incoming/                      uploaded originals (audit trail)
    data/temp/<batch-id>/               split parts while a batch is processed
    data/review/<Client>/               documents below the confidence threshold
    data/processed/Clients/<Last_First> exported, convention compliant documents

Every move is logged; nothing is ever overwritten (see
:func:`modules.renamer.unique_path`).
"""

from __future__ import annotations

import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


from config import SETTINGS, Settings, get_logger
from modules.renamer import unique_path

logger = get_logger(__name__)

_SAFE_FOLDER = re.compile(r"[^\w.\- ]+", re.UNICODE)


class StorageError(RuntimeError):
    """Raised when a file cannot be stored."""


@dataclass(frozen=True)
class ClientFolder:
    """A client folder with its documents."""

    key: str
    path: Path
    files: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def display_name(self) -> str:
        """``Smith_John`` rendered as ``Smith, John``."""
        parts = self.key.split("_", 1)
        return f"{parts[0]}, {parts[1]}" if len(parts) == 2 else self.key

    @property
    def document_count(self) -> int:
        """Number of documents currently filed for this client."""
        return len(self.files)


def safe_folder_name(value: str, fallback: str = "UNKNOWN") -> str:
    """Return a file-system safe folder name.

    Underscores are preserved: client folders are named ``LastName_FirstName``.
    """
    cleaned = _SAFE_FOLDER.sub("", str(value or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or fallback


class FolderManager:
    """Creates, resolves and populates the application's directories."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings
        self._paths = settings.paths
        self._paths.ensure()

    # -- properties -------------------------------------------------------- #
    @property
    def incoming_dir(self) -> Path:
        """Directory holding the uploaded originals."""
        return self._paths.incoming

    @property
    def review_dir(self) -> Path:
        """Directory holding documents that require a manual review."""
        return self._paths.review

    @property
    def clients_dir(self) -> Path:
        """Root of the ``Clients/`` tree."""
        return self._paths.clients

    @property
    def temp_dir(self) -> Path:
        """Root of the temporary work area."""
        return self._paths.temp

    # -- uploads and temp -------------------------------------------------- #
    def save_upload(self, filename: str, data: bytes) -> Path:
        """Persist an uploaded file in ``data/incoming`` and return its path."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = safe_folder_name(Path(filename).stem, "upload")
        target = unique_path(self.incoming_dir, f"{stamp}_{safe_name}.pdf")
        try:
            target.write_bytes(data)
        except OSError as exc:
            raise StorageError(f"Cannot store upload {filename}: {exc}") from exc
        logger.info("Upload stored: %s (%.1f KB)", target.name, len(data) / 1024)
        return target

    def new_batch_dir(self, prefix: str = "batch") -> Path:
        """Create and return a fresh working directory under ``data/temp``."""
        batch_dir = self.temp_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Created work directory %s", batch_dir)
        return batch_dir

    def cleanup_temp(self, older_than_hours: int = 24) -> int:
        """Delete temp batches older than *older_than_hours*; return the count."""
        cutoff = time.time() - older_than_hours * 3600
        removed = 0
        for child in self.temp_dir.glob("*"):
            try:
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError as exc:  # pragma: no cover - permission issues
                logger.warning("Could not clean %s: %s", child, exc)
        if removed:
            logger.info("Removed %d stale temp folder(s)", removed)
        return removed

    # -- client folders ---------------------------------------------------- #
    def client_folder(self, client_key: str, create: bool = True) -> Path:
        """Return ``Clients/<client_key>``, creating it on demand."""
        folder = self.clients_dir / safe_folder_name(client_key)
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def review_folder(self, client_key: str | None = None, create: bool = True) -> Path:
        """Return the review folder, optionally scoped to a client."""
        folder = self.review_dir / safe_folder_name(client_key) if client_key else self.review_dir
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def list_clients(self) -> list[ClientFolder]:
        """Return every client folder together with its documents."""
        clients: list[ClientFolder] = []
        if not self.clients_dir.exists():
            return clients
        for folder in sorted(self.clients_dir.iterdir()):
            if not folder.is_dir():
                continue
            files = tuple(sorted(path for path in folder.glob("*.pdf") if path.is_file()))
            clients.append(ClientFolder(key=folder.name, path=folder, files=files))
        return clients

    def list_review_files(self) -> list[Path]:
        """Return every document currently waiting in ``data/review``."""
        if not self.review_dir.exists():
            return []
        return sorted(path for path in self.review_dir.rglob("*.pdf") if path.is_file())

    # -- storing ----------------------------------------------------------- #
    def store_processed(self, source: Path, filename: str, client_key: str, *, move: bool = True) -> Path:
        """File a finished document in its client folder.

        Args:
            source: Path of the (already split) document.
            filename: Convention compliant target file name.
            client_key: ``LastName_FirstName``.
            move: Move (default) or copy the file.

        Returns:
            The final path inside ``Clients/<client_key>``.
        """
        return self._place(source, filename, self.client_folder(client_key), move=move, label="processed")

    def store_review(self, source: Path, filename: str, client_key: str | None = None, *, move: bool = True) -> Path:
        """File a low-confidence document in ``data/review``."""
        return self._place(source, filename, self.review_folder(client_key), move=move, label="review")

    def _place(self, source: Path, filename: str, folder: Path, *, move: bool, label: str) -> Path:
        """Copy or move *source* to *folder*/*filename* without overwriting."""
        source = Path(source)
        if not source.is_file():
            raise StorageError(f"Source file does not exist: {source}")
        folder.mkdir(parents=True, exist_ok=True)
        target = unique_path(folder, filename)
        try:
            if move:
                shutil.move(str(source), str(target))
            else:
                shutil.copy2(source, target)
        except OSError as exc:
            raise StorageError(f"Cannot store {filename} in {folder}: {exc}") from exc
        logger.info("Stored (%s): %s", label, target.relative_to(self._paths.base) if self._is_inside(target) else target)
        return target

    def _is_inside(self, path: Path) -> bool:
        """True when *path* lives inside the application directory."""
        try:
            path.relative_to(self._paths.base)
            return True
        except ValueError:
            return False
