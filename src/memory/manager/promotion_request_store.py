from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from ..entities import PromotionRequest, PromotionStatus


class PromotionRequestNotFoundError(Exception):
    """Raised when a promotion request cannot be found."""


class PromotionRequestAlreadyExistsError(Exception):
    """Raised when a promotion request with the same ID already exists."""


class PromotionRequestStore:
    """
    JSON-backed persistent store for PromotionRequest objects.

    Responsibilities:
    - create promotion requests
    - retrieve requests by request ID
    - replace existing requests after review or approval
    - list all requests
    - list pending requests

    This class only handles persistence. Permission checks and promotion
    business rules belong to PromotionService.
    """

    def __init__(
        self,
        file_path: str | Path = "data/promotion_requests.json",
    ) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.file_path.exists():
            self._write_raw([])

    # ------------------------------------------------------------------
    # Internal file operations
    # ------------------------------------------------------------------

    def _read_raw(self) -> list[dict]:
        """
        Read raw promotion-request dictionaries from the JSON file.
        """

        try:
            with self.file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except FileNotFoundError:
            self._write_raw([])
            return []

        except json.JSONDecodeError as error:
            raise ValueError(
                "Promotion request file contains invalid JSON: "
                f"{self.file_path}"
            ) from error

        if not isinstance(data, list):
            raise ValueError(
                "Promotion request file must contain a JSON list: "
                f"{self.file_path}"
            )

        return data

    def _write_raw(
        self,
        data: list[dict],
    ) -> None:
        """
        Write raw promotion-request dictionaries to the JSON file.

        A temporary file is used so a partially written file is less likely
        to corrupt the persistent store.
        """

        temporary_path = self.file_path.with_suffix(
            f"{self.file_path.suffix}.tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        temporary_path.replace(self.file_path)

    def _load_requests(self) -> list[PromotionRequest]:
        """
        Load and validate all stored PromotionRequest objects.
        """

        raw_requests = self._read_raw()

        try:
            return [
                PromotionRequest.model_validate(raw_request)
                for raw_request in raw_requests
            ]

        except Exception as error:
            raise ValueError(
                "Promotion request file contains an invalid request record: "
                f"{self.file_path}"
            ) from error

    def _save_requests(
        self,
        requests: list[PromotionRequest],
    ) -> None:
        """
        Serialize and save PromotionRequest objects.
        """

        raw_requests = [
            request.model_dump(mode="json")
            for request in requests
        ]

        self._write_raw(raw_requests)

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
                If a request with the same request_id already exists.
        """

        if not isinstance(request, PromotionRequest):
            raise TypeError(
                "request must be a PromotionRequest instance."
            )

        requests = self._load_requests()

        request_exists = any(
            existing_request.request_id == request.request_id
            for existing_request in requests
        )

        if request_exists:
            raise PromotionRequestAlreadyExistsError(
                "Promotion request already exists: "
                f"{request.request_id!r}."
            )

        requests.append(deepcopy(request))
        self._save_requests(requests)

        return deepcopy(request)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(
        self,
        request_id: str,
    ) -> PromotionRequest:
        """
        Retrieve a promotion request by request ID.

        Raises:
            PromotionRequestNotFoundError:
                If no request with the supplied request_id exists.
        """

        request_id = request_id.strip()

        if not request_id:
            raise ValueError(
                "request_id cannot be empty."
            )

        for request in self._load_requests():
            if request.request_id == request_id:
                return deepcopy(request)

        raise PromotionRequestNotFoundError(
            f"Promotion request {request_id!r} was not found."
        )

    def list_all(self) -> list[PromotionRequest]:
        """
        Return all promotion requests.
        """

        return deepcopy(
            self._load_requests()
        )

    def list_pending(self) -> list[PromotionRequest]:
        """
        Return promotion requests whose status is still pending.
        """

        pending_requests = [
            request
            for request in self._load_requests()
            if request.status == PromotionStatus.PENDING
        ]

        return deepcopy(pending_requests)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def replace(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        """
        Replace an existing promotion request.

        PromotionService uses this after Critic review, Coordinator approval,
        or Coordinator rejection.

        Raises:
            PromotionRequestNotFoundError:
                If the request does not already exist.
        """

        if not isinstance(request, PromotionRequest):
            raise TypeError(
                "request must be a PromotionRequest instance."
            )

        requests = self._load_requests()

        for index, existing_request in enumerate(requests):
            if existing_request.request_id == request.request_id:
                requests[index] = deepcopy(request)
                self._save_requests(requests)

                return deepcopy(request)

        raise PromotionRequestNotFoundError(
            "Cannot replace missing promotion request: "
            f"{request.request_id!r}."
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the total number of stored promotion requests.
        """

        return len(
            self._load_requests()
        )

    def clear(self) -> None:
        """
        Remove all promotion requests.

        Intended for tests and development reset.
        """

        self._write_raw([])