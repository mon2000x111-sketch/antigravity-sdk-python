# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base definitions for Antigravity SDK Hooks v2.

This module defines the interface for Hooks and the standard result types
returned by their lifecycle callbacks.
"""
from __future__ import annotations

import abc
from typing import Any, Optional

from google.antigravity.types import AskQuestionInteractionSpec
from google.antigravity.types import HookResult
from google.antigravity.types import QuestionHookResult

# --- Contexts ---


class HookContext:
  """Base context for hooks to share state."""

  def __init__(self, parent: Optional["HookContext"] = None):
    self.parent = parent
    self._store: dict[str, Any] = {}

  def get(self, key: str, default: Any = None) -> Any:
    """Gets a value from the context or its parents."""
    if key in self._store:
      return self._store[key]
    if self.parent:
      return self.parent.get(key, default)
    return default

  def set(self, key: str, value: Any) -> None:
    """Sets a value in the local context."""
    self._store[key] = value


class SessionContext(HookContext):
  """Context scoped to an entire session."""

  def __init__(self):
    super().__init__(parent=None)


class TurnContext(HookContext):
  """Context scoped to a single turn."""

  def __init__(self, session_context: SessionContext):
    super().__init__(parent=session_context)


class OperationContext(HookContext):
  """Context scoped to a specific operation (e.g. tool call)."""

  def __init__(self, turn_context: TurnContext):
    super().__init__(parent=turn_context)


# --- Base Hook Types ---


class InspectHook(abc.ABC):
  """Read-only, non-blocking hook for observability."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: Any) -> None:
    """Runs the inspection hook.

    Args:
      context: The hook context.
      data: The data to inspect (read-only).
    """
    pass


class DecideHook(abc.ABC):
  """Read-only, blocking hook for policy decisions."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: Any) -> HookResult:
    """Runs the decision hook.

    Args:
      context: The hook context.
      data: The data to make a decision on.

    Returns:
      A HookResult indicating allow/deny.
    """
    pass


class TransformHook(abc.ABC):
  """Modifying, blocking hook for data transformation."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: Any) -> Any:
    """Runs the transformation hook.

    Args:
      context: The hook context.
      data: The data to transform.

    Returns:
      The transformed data.
    """
    pass


Hook = InspectHook | DecideHook | TransformHook


# --- Concrete Hook Interfaces ---


# Session
class OnSessionStartHook(InspectHook):
  """Invoked when the session starts."""

  pass


class OnSessionEndHook(InspectHook):
  """Invoked when the session ends."""

  pass


# Turn
class PreTurnHook(DecideHook):
  """Invoked before a turn starts."""

  pass


class PostTurnHook(InspectHook):
  """Invoked after a turn ends."""

  pass


# Tool
class PreToolCallDecideHook(DecideHook):
  """Invoked before a tool call to decide if it should proceed."""

  pass


class PostToolCallHook(InspectHook):
  """Invoked after a tool call completes."""

  pass


class OnToolErrorHook(TransformHook):
  """Invoked when a tool fails, allowing for recovery or modification.

  Receives the raised exception and returns the error representation that
  the model should see. If the hook returns None, the harness uses its
  default error formatting instead.

  The hook cannot fix or retry the tool call on its own, but it can guide
  the agent toward a specific resolution.
  """

  pass


# Interaction
class OnInteractionHook(TransformHook):
  """Hook invoked when the agent needs user interaction.

  This is a superset of QuestionHook and handles all user interactions.
  """

  @abc.abstractmethod
  async def run(
      self, context: HookContext, data: AskQuestionInteractionSpec
  ) -> QuestionHookResult:
    """Runs the interaction hook.

    Args:
      context: The hook context.
      data: Specification of the interaction.

    Returns:
      The interaction result.
    """
    pass


# Compaction
class OnCompactionHook(InspectHook):
  """Invoked when a context compaction event occurs.

  Compaction is triggered by the harness when the context window exceeds the
  configured token threshold. This hook provides an observability point for
  logging, metrics, or UI notifications.
  """

  pass
