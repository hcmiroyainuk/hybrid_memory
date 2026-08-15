from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import ValidationError

from ..entities import (
    PromotionRequest,
    PromotionStatus,
)


class PromotionRequestStoreError(Exception):
    """Base exception for promotion-request persistence errors."""


class PromotionRequestNotFoundError(PromotionRequestStoreError):
    """Raised when a promotion request cannot be found."""


class PromotionRequestAlreadyExistsError(PromotionRequestStoreError):
    """Raised when creating a request whose request_id already exists."""


class PromotionRequestStoreDataError(PromotionRequestStoreError):
    """Raised when the JSON file contains invalid or incompatible data."""


class PromotionRequestStore:
    """
    JSON-backed store for PromotionRequest entities.

    The store only handles persistence and lookup. It does not perform Critic
    review, Coordinator decisions, memory promotion, or ACL updates.

    Primary write methods:
        create(request): insert a new request.
        replace(request): replace an existing request.

    Compatibility methods:
        get(request_id): alias of get_by_id().
        save(request): upsert-style compatibility method.

    Returned objects are deep copies, so callers must explicitly call
    replace() or save() to persist mutations.
    """

    def __init__(
        self,
        file_path: str | Path = "data/promotion_requests.json",
    ) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

        if not self.file_path.exists():
            self._write_raw([])

    # ------------------------------------------------------------------
    # Internal file operations
    # ------------------------------------------------------------------

    def _read_raw(self) -> list[dict[str, Any]]:
        """Read and validate the top-level JSON structure."""
        with self._lock:
            try:
                with self.file_path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except FileNotFoundError:
                self._write_raw([])
                return []
            except json.JSONDecodeError as error:
                raise PromotionRequestStoreDataError(
                    "Promotion request file contains invalid JSON: "
                    f"{self.file_path}"
                ) from error
            except OSError as error:
                raise PromotionRequestStoreError(
                    "Failed to read promotion request file: "
                    f"{self.file_path}"
                ) from error

        if not isinstance(data, list):
            raise PromotionRequestStoreDataError(
                "Promotion request file must contain a JSON list: "
                f"{self.file_path}"
            )

        return data

    def _write_raw(self, data: list[dict[str, Any]]) -> None:
        """
        Atomically write raw request dictionaries.

        A temporary file is written in the same directory and moved over the
        target file with os.replace().
        """
        temporary_path = self.file_path.with_name(
            f".{self.file_path.name}.tmp"
        )

        with self._lock:
            try:
                with temporary_path.open("w", encoding="utf-8") as file:
                    json.dump(
                        data,
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
                    file.flush()
                    os.fsync(file.fileno())

                os.replace(temporary_path, self.file_path)
            except OSError as error:
                try:
                    if temporary_path.exists():
                        temporary_path.unlink()
                except OSError:
                    pass

                raise PromotionRequestStoreError(
                    "Failed to write promotion request file: "
                    f"{self.file_path}"
                ) from error

    def _load_requests(self) -> list[PromotionRequest]:
        """Load and validate all persisted requests."""
        raw_requests = self._read_raw()
        requests: list[PromotionRequest] = []

        for index, raw_request in enumerate(raw_requests):
            try:
                request = PromotionRequest.model_validate(raw_request)
            except ValidationError as error:
                raise PromotionRequestStoreDataError(
                    "Invalid PromotionRequest record at "
                    f"index {index} in {self.file_path}."
                ) from error

            requests.append(request)

        self._assert_unique_request_ids(requests)
        return requests

    def _save_requests(
        self,
        requests: list[PromotionRequest],
    ) -> None:
        """Validate and persist a complete request collection."""
        self._assert_unique_request_ids(requests)

        raw_requests = [
            request.model_dump(mode="json")
            for request in requests
        ]
        self._write_raw(raw_requests)

    @staticmethod
    def _assert_unique_request_ids(
        requests: list[PromotionRequest],
    ) -> None:
        seen: set[str] = set()

        for request in requests:
            if request.request_id in seen:
                raise PromotionRequestStoreDataError(
                    "Duplicate promotion request ID found: "
                    f"{request.request_id!r}."
                )

            seen.add(request.request_id)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        """
        Persist a new promotion request.

        Raises:
            PromotionRequestAlreadyExistsError:
                If request.request_id already exists.
        """
        self._require_request(request)

        with self._lock:
            requests = self._load_requests()

            if any(
                existing.request_id == request.request_id
                for existing in requests
            ):
                raise PromotionRequestAlreadyExistsError(
                    "Promotion request already exists: "
                    f"{request.request_id!r}."
                )

            stored = request.model_copy(deep=True)
            requests.append(stored)
            self._save_requests(requests)

        return stored.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, request_id: str) -> PromotionRequest:
        """Return a request by request_id."""
        clean_request_id = self._required_text(
            request_id,
            "request_id",
        )

        for request in self._load_requests():
            if request.request_id == clean_request_id:
                return request.model_copy(deep=True)

        raise PromotionRequestNotFoundError(
            "Promotion request was not found: "
            f"{clean_request_id!r}."
        )

    def get(self, request_id: str) -> PromotionRequest:
        """Compatibility alias for get_by_id()."""
        return self.get_by_id(request_id)

    def exists(self, request_id: str) -> bool:
        """Check whether a request exists."""
        clean_request_id = self._required_text(
            request_id,
            "request_id",
        )

        return any(
            request.request_id == clean_request_id
            for request in self._load_requests()
        )

    def list_all(self) -> list[PromotionRequest]:
        """Return all requests in creation order."""
        return deepcopy(self._load_requests())

    def list_by_status(
        self,
        status: PromotionStatus | str,
    ) -> list[PromotionRequest]:
        """Return requests with the supplied lifecycle status."""
        clean_status = PromotionStatus(status)

        return deepcopy(
            [
                request
                for request in self._load_requests()
                if request.status == clean_status
            ]
        )

    def list_pending(self) -> list[PromotionRequest]:
        return self.list_by_status(PromotionStatus.PENDING)

    def list_reviewed(self) -> list[PromotionRequest]:
        return self.list_by_status(PromotionStatus.REVIEWED)

    def list_approved(self) -> list[PromotionRequest]:
        return self.list_by_status(PromotionStatus.APPROVED)

    def list_rejected(self) -> list[PromotionRequest]:
        return self.list_by_status(PromotionStatus.REJECTED)

    def list_by_memory_id(
        self,
        memory_id: str,
    ) -> list[PromotionRequest]:
        """Return all requests concerning one memory."""
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )

        return deepcopy(
            [
                request
                for request in self._load_requests()
                if request.memory_id == clean_memory_id
            ]
        )

    def list_by_owner_agent_id(
        self,
        owner_agent_id: str,
    ) -> list[PromotionRequest]:
        """Return requests for memories owned by one agent."""
        clean_agent_id = self._required_text(
            owner_agent_id,
            "owner_agent_id",
        )

        return deepcopy(
            [
                request
                for request in self._load_requests()
                if request.owner_agent_id == clean_agent_id
            ]
        )

    def list_by_requester_agent_id(
        self,
        requester_agent_id: str,
    ) -> list[PromotionRequest]:
        """Return requests submitted by one requesting agent."""
        clean_agent_id = self._required_text(
            requester_agent_id,
            "requester_agent_id",
        )

        return deepcopy(
            [
                request
                for request in self._load_requests()
                if request.requester_agent_id == clean_agent_id
            ]
        )

    def list_by_task_id(
        self,
        task_id: str,
    ) -> list[PromotionRequest]:
        """Return requests triggered by one task."""
        clean_task_id = self._required_text(
            task_id,
            "task_id",
        )

        return deepcopy(
            [
                request
                for request in self._load_requests()
                if request.task_id == clean_task_id
            ]
        )

    def list_by_critic_agent_id(
        self,
        critic_agent_id: str,
    ) -> list[PromotionRequest]:
        """Return requests reviewed by one Critic."""
        clean_agent_id = self._required_text(
            critic_agent_id,
            "critic_agent_id",
        )

        return deepcopy(
            [
                request
                for request in self._load_requests()
                if request.critic_agent_id == clean_agent_id
            ]
        )

    def list_by_decider_agent_id(
        self,
        decided_by_agent_id: str,
    ) -> list[PromotionRequest]:
        """Return requests decided by one Coordinator."""
        clean_agent_id = self._required_text(
            decided_by_agent_id,
            "decided_by_agent_id",
        )

        return deepcopy(
            [
                request
                for request in self._load_requests()
                if request.decided_by_agent_id == clean_agent_id
            ]
        )

    def find_open_request(
        self,
        *,
        memory_id: str,
        requester_agent_id: str,
    ) -> PromotionRequest | None:
        """
        Find a pending or reviewed request for the same requester and memory.

        This allows PromotionService to avoid duplicate active requests.
        Completed requests are ignored.
        """
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )
        clean_requester_id = self._required_text(
            requester_agent_id,
            "requester_agent_id",
        )

        open_statuses = {
            PromotionStatus.PENDING,
            PromotionStatus.REVIEWED,
        }

        for request in reversed(self._load_requests()):
            if (
                request.memory_id == clean_memory_id
                and request.requester_agent_id == clean_requester_id
                and request.status in open_statuses
            ):
                return request.model_copy(deep=True)

        return None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def replace(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        """
        Replace an existing request with the same request_id.

        Raises:
            PromotionRequestNotFoundError:
                If request.request_id does not exist.
        """
        self._require_request(request)

        with self._lock:
            requests = self._load_requests()

            for index, existing in enumerate(requests):
                if existing.request_id == request.request_id:
                    stored = request.model_copy(deep=True)
                    requests[index] = stored
                    self._save_requests(requests)
                    return stored.model_copy(deep=True)

        raise PromotionRequestNotFoundError(
            "Promotion request was not found: "
            f"{request.request_id!r}."
        )

    def save(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        """
        Upsert-style compatibility method.

        New service code should prefer create() for submission and replace()
        for Critic review or Coordinator decisions.
        """
        self._require_request(request)

        if self.exists(request.request_id):
            return self.replace(request)

        return self.create(request)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of persisted requests."""
        return len(self._load_requests())

    def clear(self) -> None:
        """
        Clear all requests.

        Intended only for tests, demos, or isolated experiment reset.
        """
        self._write_raw([])

    def export_as_dicts(self) -> list[dict[str, Any]]:
        """Export requests as JSON-compatible dictionaries."""
        return [
            request.model_dump(mode="json")
            for request in self._load_requests()
        ]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_request(
        request: PromotionRequest,
    ) -> None:
        if not isinstance(request, PromotionRequest):
            raise TypeError(
                "request must be a PromotionRequest instance."
            )

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
    ) -> str:
        cleaned = str(value or "").strip()

        if not cleaned:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return cleaned


__all__ = [
    "PromotionRequestStoreError",
    "PromotionRequestNotFoundError",
    "PromotionRequestAlreadyExistsError",
    "PromotionRequestStoreDataError",
    "PromotionRequestStore",
]