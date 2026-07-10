from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

from ..entities import (
    MemoryOperationRecord,
    MemoryOperationType,
)


class OperationLogNotFoundError(Exception):
    """Raised when an operation log record cannot be found."""


class OperationLogAlreadyExistsError(Exception):
    """Raised when trying to append an operation log with an existing record_id."""


class OperationLogStore:
    """
    JSON-backed append-oriented store for MemoryOperationRecord.

    This store is used as an audit log for memory governance operations.

    It supports:
    - append operation record
    - get operation by record_id
    - list all operations
    - list operations by memory_id
    - list operations by agent_id
    - list operations by operation_type
    - list operations by related_request_id
    - clear logs for testing/demo reset

    It intentionally does not provide business-level update or delete methods,
    because operation logs should behave like an audit trail.
    """

    def __init__(self, file_path: str | Path = "data/operation_logs.json") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists():
            self._write_raw([])

    # -------------------------
    # Internal file operations
    # -------------------------

    def _read_raw(self) -> list[dict]:
        """
        Read raw log records from the JSON file.
        """
        try:
            with self.file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(
                f"Operation log file '{self.file_path}' contains invalid JSON."
            )

        if not isinstance(data, list):
            raise ValueError(
                f"Operation log file '{self.file_path}' must contain a JSON list."
            )

        return data

    def _write_raw(self, data: list[dict]) -> None:
        """
        Write raw log records to the JSON file.
        """
        with self.file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    def _load_records(self) -> list[MemoryOperationRecord]:
        """
        Load all records as MemoryOperationRecord objects.
        """
        raw_records = self._read_raw()
        return [
            MemoryOperationRecord.model_validate(record)
            for record in raw_records
        ]

    def _save_records(self, records: list[MemoryOperationRecord]) -> None:
        """
        Save MemoryOperationRecord objects to the JSON file.
        """
        raw_records = [
            record.model_dump(mode="json")
            for record in records
        ]
        self._write_raw(raw_records)

    # -------------------------
    # Create / Append
    # -------------------------

    def append(self, record: MemoryOperationRecord) -> MemoryOperationRecord:
        """
        Append a new operation record.

        Raises:
            OperationLogAlreadyExistsError:
                If a record with the same record_id already exists.
        """
        records = self._load_records()

        if any(existing.record_id == record.record_id for existing in records):
            raise OperationLogAlreadyExistsError(
                f"Operation log with id '{record.record_id}' already exists."
            )

        records.append(deepcopy(record))
        self._save_records(records)

        return deepcopy(record)

    # -------------------------
    # Read
    # -------------------------

    def get_by_id(self, record_id: str) -> MemoryOperationRecord:
        """
        Get an operation record by record_id.

        Raises:
            OperationLogNotFoundError:
                If the record does not exist.
        """
        records = self._load_records()

        for record in records:
            if record.record_id == record_id:
                return deepcopy(record)

        raise OperationLogNotFoundError(
            f"Operation log with id '{record_id}' was not found."
        )

    def exists(self, record_id: str) -> bool:
        """
        Check whether an operation record exists.
        """
        records = self._load_records()
        return any(record.record_id == record_id for record in records)

    def list_all(self) -> list[MemoryOperationRecord]:
        """
        Return all operation records.
        """
        return deepcopy(self._load_records())

    def list_by_memory_id(self, memory_id: str) -> list[MemoryOperationRecord]:
        """
        Return all operation records related to a specific memory_id.

        A record may target one or multiple memories, so this checks whether
        memory_id appears in target_memory_ids.
        """
        records = self._load_records()

        result = [
            record
            for record in records
            if memory_id in record.target_memory_ids
        ]

        return deepcopy(result)

    def list_by_agent_id(self, agent_id: str) -> list[MemoryOperationRecord]:
        """
        Return operation records performed by or reviewed by a specific agent.
        """
        records = self._load_records()

        result = [
            record
            for record in records
            if record.actor_agent_id == agent_id
            or record.reviewer_agent_id == agent_id
        ]

        return deepcopy(result)

    def list_by_actor_id(self, actor_agent_id: str) -> list[MemoryOperationRecord]:
        """
        Return operation records performed by a specific actor agent.
        """
        records = self._load_records()

        result = [
            record
            for record in records
            if record.actor_agent_id == actor_agent_id
        ]

        return deepcopy(result)

    def list_by_reviewer_id(self, reviewer_agent_id: str) -> list[MemoryOperationRecord]:
        """
        Return operation records reviewed by a specific reviewer agent.
        """
        records = self._load_records()

        result = [
            record
            for record in records
            if record.reviewer_agent_id == reviewer_agent_id
        ]

        return deepcopy(result)

    def list_by_operation_type(
        self,
        operation_type: MemoryOperationType | str,
    ) -> list[MemoryOperationRecord]:
        """
        Return operation records by operation type.
        """
        operation_type = MemoryOperationType(operation_type)

        records = self._load_records()

        result = [
            record
            for record in records
            if record.operation_type == operation_type
        ]

        return deepcopy(result)

    def list_by_request_id(self, request_id: str) -> list[MemoryOperationRecord]:
        """
        Return operation records related to a promotion request.
        """
        records = self._load_records()

        result = [
            record
            for record in records
            if record.related_request_id == request_id
        ]

        return deepcopy(result)

    def list_by_status(self, status: str) -> list[MemoryOperationRecord]:
        """
        Return operation records by operation status.

        This uses string comparison to avoid importing OperationStatus unless needed.
        """
        records = self._load_records()

        result = [
            record
            for record in records
            if record.status == status
        ]

        return deepcopy(result)

    # -------------------------
    # Utility
    # -------------------------

    def count(self) -> int:
        """
        Return the number of operation records.
        """
        return len(self._load_records())

    def clear(self) -> None:
        """
        Clear all operation records.

        Intended for tests or demo reset only.
        Formal business logic should not delete audit logs.
        """
        self._write_raw([])

    def export_as_dicts(self) -> list[dict]:
        """
        Export all records as plain dictionaries.

        Useful for debugging, tests, or displaying JSON data.
        """
        return [
            record.model_dump(mode="json")
            for record in self._load_records()
        ]