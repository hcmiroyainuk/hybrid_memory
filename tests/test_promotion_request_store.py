import pytest

from src.memory.entities import (
    PromotionRequest,
    PromotionStatus,
)
from src.memory.manager import (
    PromotionRequestStore,
    PromotionRequestNotFoundError,
    PromotionRequestAlreadyExistsError,
)


@pytest.fixture
def request_store(tmp_path) -> PromotionRequestStore:
    return PromotionRequestStore(
        tmp_path / "promotion_requests.json"
    )


@pytest.fixture
def promotion_request() -> PromotionRequest:
    return PromotionRequest(
        memory_id="memory_001",
        proposed_by_agent_id="worker_b",
        reason="The memory passed conflict checking.",
    )


def test_create_and_get_request(
    request_store: PromotionRequestStore,
    promotion_request: PromotionRequest,
) -> None:
    created_request = request_store.create(
        promotion_request
    )

    loaded_request = request_store.get_by_id(
        created_request.request_id
    )

    assert loaded_request.request_id == created_request.request_id
    assert loaded_request.memory_id == "memory_001"
    assert loaded_request.status == PromotionStatus.PENDING
    assert request_store.count() == 1


def test_create_rejects_duplicate_request_id(
    request_store: PromotionRequestStore,
    promotion_request: PromotionRequest,
) -> None:
    request_store.create(promotion_request)

    with pytest.raises(
        PromotionRequestAlreadyExistsError
    ):
        request_store.create(promotion_request)


def test_get_missing_request_raises_error(
    request_store: PromotionRequestStore,
) -> None:
    with pytest.raises(
        PromotionRequestNotFoundError
    ):
        request_store.get_by_id(
            "missing_request"
        )


def test_replace_request(
    request_store: PromotionRequestStore,
    promotion_request: PromotionRequest,
) -> None:
    created_request = request_store.create(
        promotion_request
    )

    created_request.approve(
        reviewer_agent_id="coordinator",
        comment="Approved for shared access.",
    )

    updated_request = request_store.replace(
        created_request
    )

    loaded_request = request_store.get_by_id(
        updated_request.request_id
    )

    assert loaded_request.status == PromotionStatus.APPROVED
    assert loaded_request.reviewed_by_agent_id == "coordinator"
    assert loaded_request.review_comment == (
        "Approved for shared access."
    )


def test_replace_missing_request_raises_error(
    request_store: PromotionRequestStore,
    promotion_request: PromotionRequest,
) -> None:
    with pytest.raises(
        PromotionRequestNotFoundError
    ):
        request_store.replace(
            promotion_request
        )


def test_list_all_requests(
    request_store: PromotionRequestStore,
) -> None:
    first_request = PromotionRequest(
        memory_id="memory_001",
        proposed_by_agent_id="worker_a",
        reason="First request.",
    )

    second_request = PromotionRequest(
        memory_id="memory_002",
        proposed_by_agent_id="worker_b",
        reason="Second request.",
    )

    request_store.create(first_request)
    request_store.create(second_request)

    requests = request_store.list_all()

    assert len(requests) == 2
    assert {
        request.memory_id
        for request in requests
    } == {
        "memory_001",
        "memory_002",
    }


def test_list_pending_requests(
    request_store: PromotionRequestStore,
) -> None:
    pending_request = PromotionRequest(
        memory_id="memory_pending",
        proposed_by_agent_id="worker_a",
        reason="Pending request.",
    )

    approved_request = PromotionRequest(
        memory_id="memory_approved",
        proposed_by_agent_id="worker_b",
        reason="Approved request.",
    )

    approved_request.approve(
        reviewer_agent_id="coordinator",
        comment="Approved.",
    )

    request_store.create(pending_request)
    request_store.create(approved_request)

    pending_requests = request_store.list_pending()

    assert len(pending_requests) == 1
    assert (
        pending_requests[0].request_id
        == pending_request.request_id
    )


def test_requests_persist_between_store_instances(
    tmp_path,
    promotion_request: PromotionRequest,
) -> None:
    file_path = tmp_path / "promotion_requests.json"

    first_store = PromotionRequestStore(file_path)
    first_store.create(promotion_request)

    second_store = PromotionRequestStore(file_path)

    loaded_request = second_store.get_by_id(
        promotion_request.request_id
    )

    assert (
        loaded_request.request_id
        == promotion_request.request_id
    )
    assert loaded_request.memory_id == "memory_001"


def test_clear_requests(
    request_store: PromotionRequestStore,
    promotion_request: PromotionRequest,
) -> None:
    request_store.create(promotion_request)

    request_store.clear()

    assert request_store.count() == 0
    assert request_store.list_all() == []