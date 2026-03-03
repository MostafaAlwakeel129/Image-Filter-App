"""
Global Undo Manager for the Computer Vision Suite.

Usage
-----
from Helpers.undo_manager import UndoManager

# Push a state before applying an operation:
UndoManager.push(proc_label_bytes, status_text, restore_callback)

# Undo the last operation:
UndoManager.undo()

# Connect the global button/shortcut:
UndoManager.set_button(btn)          # keeps button enabled/disabled in sync
"""
from __future__ import annotations
from typing import Callable, Any


class _UndoManager:
    MAX_HISTORY = 20

    def __init__(self):
        self._stack: list[dict] = []   # list of {bytes, label, callback}
        self._button = None            # global QPushButton reference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        image_bytes: bytes | None,
        status_text: str,
        restore_fn: Callable[[bytes | None, str], None],
    ):
        """
        Save a snapshot that can be restored later.

        Parameters
        ----------
        image_bytes : bytes | None
            The processed image bytes BEFORE the new operation is applied.
        status_text : str
            The status bar text BEFORE the new operation is applied.
        restore_fn : callable(image_bytes, status_text)
            A tab-specific function that knows how to restore both the
            processed image panel and the status label from a snapshot.
        """
        self._stack.append({
            "bytes":   image_bytes,
            "status":  status_text,
            "restore": restore_fn,
        })
        if len(self._stack) > self.MAX_HISTORY:
            self._stack.pop(0)
        self._sync_button()

    def undo(self):
        """Restore the most recent snapshot and remove it from the stack."""
        if not self._stack:
            return
        snap = self._stack.pop()
        snap["restore"](snap["bytes"], snap["status"])
        self._sync_button()

    def clear(self):
        """Discard all history (e.g. when a new image is loaded)."""
        self._stack.clear()
        self._sync_button()

    def set_button(self, btn):
        """Register the global Undo QPushButton so its state stays in sync."""
        self._button = btn
        self._sync_button()

    @property
    def can_undo(self) -> bool:
        return bool(self._stack)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sync_button(self):
        if self._button is not None:
            self._button.setEnabled(self.can_undo)
            tip = f"Undo  ({len(self._stack)} steps)" if self.can_undo else "Nothing to undo"
            self._button.setToolTip(tip)


# Singleton instance used by all modules
UndoManager = _UndoManager()