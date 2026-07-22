"""Versioned server-to-runner session initialization payloads."""

from __future__ import annotations

import base64
import binascii
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omnigent.entities import Conversation

SESSION_INIT_PROTOCOL_VERSION = 2
SESSION_INIT_PAYLOAD_KEY = "session_init"
# Keep the base64-expanded JSON comfortably below the tunnel's 100 MiB frame
# limit while allowing substantially larger bundles than normal agents use.
MAX_EMBEDDED_AGENT_BUNDLE_BYTES = 16 * 1024 * 1024
_MAX_EMBEDDED_AGENT_BUNDLE_BASE64_CHARS = 4 * ((MAX_EMBEDDED_AGENT_BUNDLE_BYTES + 2) // 3)


class RunnerSessionInitAgentBundle(BaseModel):
    """Agent archive embedded in the server-to-runner initialization request."""

    model_config = ConfigDict(extra="ignore")

    version: str
    name: str
    session_scoped: bool
    contents_base64: str


class RunnerSessionInitSnapshot(BaseModel):
    """Server-owned session state needed while starting a runner session."""

    model_config = ConfigDict(extra="ignore")

    created_at: int
    updated_at: int
    workspace: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    reasoning_effort: str | None = None
    model_override: str | None = None
    harness_override: str | None = None
    cost_control_mode_override: str | None = None
    terminal_launch_args: list[str] | None = None
    external_session_id: str | None = None
    parent_session_id: str | None = None
    root_session_id: str | None = None


class RunnerSessionInitEnvelope(BaseModel):
    """Metadata a current server can send instead of runner callback reads."""

    model_config = ConfigDict(extra="ignore")

    protocol_version: Literal[SESSION_INIT_PROTOCOL_VERSION]
    server_version: str
    session_id: str
    agent_id: str
    sub_agent_name: str | None = None
    snapshot: RunnerSessionInitSnapshot
    agent_bundle: RunnerSessionInitAgentBundle | None = None


def encode_runner_session_init_agent_bundle(
    contents: bytes,
    *,
    version: str,
    name: str,
    session_scoped: bool,
) -> RunnerSessionInitAgentBundle | None:
    """Encode a bundle when it is small enough for the initialization envelope."""
    if len(contents) > MAX_EMBEDDED_AGENT_BUNDLE_BYTES:
        return None
    return RunnerSessionInitAgentBundle(
        version=version,
        name=name,
        session_scoped=session_scoped,
        contents_base64=base64.b64encode(contents).decode("ascii"),
    )


def decode_runner_session_init_agent_bundle(
    bundle: RunnerSessionInitAgentBundle,
) -> bytes:
    """Decode and size-check an initialization bundle from an untrusted peer."""
    encoded = bundle.contents_base64
    if len(encoded) > _MAX_EMBEDDED_AGENT_BUNDLE_BASE64_CHARS:
        raise ValueError("embedded agent bundle exceeds the size limit")
    try:
        contents = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("embedded agent bundle is not valid base64") from exc
    if len(contents) > MAX_EMBEDDED_AGENT_BUNDLE_BYTES:
        raise ValueError("embedded agent bundle exceeds the size limit")
    return contents


def build_runner_session_init_payload(
    conversation: Conversation,
    *,
    server_version: str,
    agent_bundle: RunnerSessionInitAgentBundle | None = None,
) -> dict[str, Any]:
    """Build the versioned initialization fields appended to the legacy body."""
    if conversation.agent_id is None:
        raise ValueError("runner session initialization requires an agent_id")
    envelope = RunnerSessionInitEnvelope(
        protocol_version=SESSION_INIT_PROTOCOL_VERSION,
        server_version=server_version,
        session_id=conversation.id,
        agent_id=conversation.agent_id,
        sub_agent_name=conversation.sub_agent_name,
        agent_bundle=agent_bundle,
        snapshot=RunnerSessionInitSnapshot(
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            workspace=conversation.workspace,
            labels=conversation.labels,
            reasoning_effort=conversation.reasoning_effort,
            model_override=conversation.model_override,
            harness_override=conversation.harness_override,
            cost_control_mode_override=conversation.cost_control_mode_override,
            terminal_launch_args=conversation.terminal_launch_args,
            external_session_id=conversation.external_session_id,
            parent_session_id=conversation.parent_conversation_id,
            root_session_id=conversation.root_conversation_id,
        ),
    )
    return {
        "session_id": conversation.id,
        "agent_id": conversation.agent_id,
        "sub_agent_name": conversation.sub_agent_name,
        SESSION_INIT_PAYLOAD_KEY: envelope.model_dump(mode="json"),
    }


def parse_runner_session_init_envelope(
    body: dict[str, Any],
) -> RunnerSessionInitEnvelope | None:
    """Return a supported envelope, or ``None`` for the removable legacy path."""
    raw = body.get(SESSION_INIT_PAYLOAD_KEY)
    if not isinstance(raw, dict):
        return None
    if raw.get("protocol_version") != SESSION_INIT_PROTOCOL_VERSION:
        return None
    try:
        return RunnerSessionInitEnvelope.model_validate(raw)
    except ValidationError as exc:
        raise ValueError("invalid runner session initialization envelope") from exc
