from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.adapters import (
    MemoryAccessDecision,
    MemoryAccessStatus,
    MemorySharingConflictError,
    MemorySharingGateway,
    MemorySharingNotFoundError,
    MemorySharingPermissionError,
    PromotionMemorySharingAdapter,
)
from src.memory.services.promotion_service import (
    AccessAlreadyGrantedError,
    InvalidPromotionActorError,
    MemoryNotFoundError,
    PromotionRequestNotFoundError,
    PromotionStateError,
)


class FakePromotionService:
    """Narrow fake used to test adapter behaviour independently of stores."""

    def __init__(self) -> None:
        self.requests: dict[str, SimpleNamespace] = {}
        self.access: set[tuple[str, str]] = set()
        self.next_error: Exception | None = None
        self.calls: list[tuple[str, dict]] = []

    def _raise_if_configured(self) -> None:
        if self.next_error is None:
            return

        error = self.next_error
        self.next_error = None
        raise error

    def has_access(self, *, memory_id: str, agent_id: str) -> bool:
        self._raise_if_configured()
        self.calls.append(
            (
                "has_access",
                {"memory_id": memory_id, "agent_id": agent_id},
            )
        )
        return (memory_id, agent_id) in self.access

    def submit_promotion_request(
        self,
        *,
        requester,
        memory_id: str,
        reason: str,
        task_id: str | None = None,
    ) -> SimpleNamespace:
        self._raise_if_configured()
        self.calls.append(
            (
                "submit_promotion_request",
                {
                    "requester": requester,
                    "memory_id": memory_id,
                    "reason": reason,
                    "task_id": task_id,
                },
            )
        )

        request = SimpleNamespace(
            request_id="request_001",
            memory_id=memory_id,
            owner_agent_id="alice_agent",
            requester_agent_id=requester.agent_id,
            task_id=task_id,
            status="pending",
            critic_recommendation=None,
        )
        self.requests[request.request_id] = request
        return request

    def review_promotion_request(
        self,
        *,
        critic,
        request_id: str,
        recommendation: str,
        comment: str | None = None,
    ) -> SimpleNamespace:
        self._raise_if_configured()
        self.calls.append(
            (
                "review_promotion_request",
                {
                    "critic": critic,
                    "request_id": request_id,
                    "recommendation": recommendation,
                    "comment": comment,
                },
            )
        )

        request = self.requests[request_id]
        request.status = "reviewed"
        request.critic_recommendation = recommendation
        return request

    def approve_promotion_request(
        self,
        *,
        coordinator,
        request_id: str,
        comment: str | None = None,
        require_critic_review: bool = True,
    ) -> SimpleNamespace:
        self._raise_if_configured()
        self.calls.append(
            (
                "approve_promotion_request",
                {
                    "coordinator": coordinator,
                    "request_id": request_id,
                    "comment": comment,
                    "require_critic_review": require_critic_review,
                },
            )
        )

        request = self.requests[request_id]
        request.status = "approved"
        self.access.add((request.memory_id, request.requester_agent_id))
        return request

    def approve_ungoverned_request(
        self,
        *,
        coordinator,
        request_id: str,
        comment: str | None = None,
    ) -> SimpleNamespace:
        self._raise_if_configured()
        self.calls.append(
            (
                "approve_ungoverned_request",
                {
                    "coordinator": coordinator,
                    "request_id": request_id,
                    "comment": comment,
                },
            )
        )

        request = self.requests[request_id]
        request.status = "approved"
        self.access.add((request.memory_id, request.requester_agent_id))
        return request

    def reject_promotion_request(
        self,
        *,
        coordinator,
        request_id: str,
        comment: str | None = None,
        require_critic_review: bool = True,
    ) -> SimpleNamespace:
        self._raise_if_configured()
        self.calls.append(
            (
                "reject_promotion_request",
                {
                    "coordinator": coordinator,
                    "request_id": request_id,
                    "comment": comment,
                    "require_critic_review": require_critic_review,
                },
            )
        )

        request = self.requests[request_id]
        request.status = "rejected"
        return request

    def get_request(self, request_id: str) -> SimpleNamespace:
        self._raise_if_configured()
        self.calls.append(("get_request", {"request_id": request_id}))
        return self.requests[request_id]


@pytest.fixture
def agents() -> dict[str, SimpleNamespace]:
    return {
        "alice_agent": SimpleNamespace(
            agent_id="alice_agent",
            role="worker",
        ),
        "bob_agent": SimpleNamespace(
            agent_id="bob_agent",
            role="worker",
        ),
        "critic_agent": SimpleNamespace(
            agent_id="critic_agent",
            role="critic",
        ),
        "coordinator_agent": SimpleNamespace(
            agent_id="coordinator_agent",
            role="coordinator",
        ),
    }


@pytest.fixture
def service() -> FakePromotionService:
    return FakePromotionService()


@pytest.fixture
def adapter(
    service: FakePromotionService,
    agents: dict[str, SimpleNamespace],
) -> PromotionMemorySharingAdapter:
    return PromotionMemorySharingAdapter(
        promotion_service=service,  # type: ignore[arg-type]
        agents=agents,  # type: ignore[arg-type]
    )


def test_adapter_conforms_to_gateway_protocol(adapter) -> None:
    assert isinstance(adapter, MemorySharingGateway)


def test_request_and_review_are_converted_to_dto(adapter) -> None:
    submitted = adapter.request_access(
        requester_agent_id="bob_agent",
        memory_id="memory_001",
        reason="Need the owner's information.",
        task_id="task_001",
    )

    assert submitted.request_id == "request_001"
    assert submitted.owner_agent_id == "alice_agent"
    assert submitted.requester_agent_id == "bob_agent"
    assert submitted.status == MemoryAccessStatus.PENDING

    reviewed = adapter.review_request(
        critic_agent_id="critic_agent",
        request_id=submitted.request_id,
        recommendation="approve",
        comment="Relevant and policy-compliant.",
    )

    assert reviewed.status == MemoryAccessStatus.REVIEWED
    assert reviewed.critic_recommendation == "approve"


def test_governed_approval_grants_access(adapter) -> None:
    submitted = adapter.request_access(
        requester_agent_id="bob_agent",
        memory_id="memory_001",
        reason="Required for the task.",
    )
    adapter.review_request(
        critic_agent_id="critic_agent",
        request_id=submitted.request_id,
        recommendation="approve",
    )

    decision = adapter.approve_request(
        coordinator_agent_id="coordinator_agent",
        request_id=submitted.request_id,
    )

    assert decision.decision == MemoryAccessDecision.APPROVED
    assert decision.access_granted is True
    assert adapter.has_access(
        agent_id="bob_agent",
        memory_id="memory_001",
    )


def test_direct_approval_uses_ungoverned_path(adapter, service) -> None:
    submitted = adapter.request_access(
        requester_agent_id="bob_agent",
        memory_id="memory_001",
        reason="Required for the task.",
    )

    decision = adapter.approve_direct_request(
        coordinator_agent_id="coordinator_agent",
        request_id=submitted.request_id,
    )

    assert decision.access_granted is True
    assert service.calls[-1][0] == "approve_ungoverned_request"


def test_rejection_does_not_grant_access(adapter) -> None:
    submitted = adapter.request_access(
        requester_agent_id="bob_agent",
        memory_id="memory_001",
        reason="Required for the task.",
    )
    adapter.review_request(
        critic_agent_id="critic_agent",
        request_id=submitted.request_id,
        recommendation="reject",
    )

    decision = adapter.reject_request(
        coordinator_agent_id="coordinator_agent",
        request_id=submitted.request_id,
    )

    assert decision.decision == MemoryAccessDecision.REJECTED
    assert decision.access_granted is False
    assert not adapter.has_access(
        agent_id="bob_agent",
        memory_id="memory_001",
    )


def test_unknown_agent_is_gateway_not_found(adapter) -> None:
    with pytest.raises(MemorySharingNotFoundError):
        adapter.request_access(
            requester_agent_id="unknown_agent",
            memory_id="memory_001",
            reason="Required for the task.",
        )


@pytest.mark.parametrize(
    ("service_error", "gateway_error"),
    [
        (
            PromotionRequestNotFoundError("request missing"),
            MemorySharingNotFoundError,
        ),
        (
            MemoryNotFoundError("memory missing"),
            MemorySharingNotFoundError,
        ),
        (
            AccessAlreadyGrantedError("already granted"),
            MemorySharingConflictError,
        ),
        (
            PromotionStateError("invalid state"),
            MemorySharingConflictError,
        ),
        (
            InvalidPromotionActorError("invalid actor"),
            MemorySharingPermissionError,
        ),
    ],
)
def test_service_exceptions_are_translated(
    adapter,
    service,
    service_error,
    gateway_error,
) -> None:
    service.next_error = service_error

    with pytest.raises(gateway_error):
        adapter.has_access(
            agent_id="bob_agent",
            memory_id="memory_001",
        )