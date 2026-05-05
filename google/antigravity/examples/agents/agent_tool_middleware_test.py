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

"""Tests for agent_tool_middleware hooks.

Shows how to test each hook type in isolation — no model, no agent,
no network. Each test instantiates the hook, feeds it a ToolCall or
error, and asserts the output. This is the recommended pattern for
verifying hook behavior before wiring it into an agent.
"""

import unittest

from google.antigravity import types
from google.antigravity.examples.agents import agent_tool_middleware as m
from google.antigravity.hooks import hooks


def _ctx() -> hooks.HookContext:
  """Creates a minimal HookContext for testing.

  Returns:
    A HookContext suitable for unit tests.
  """
  return hooks.OperationContext(hooks.TurnContext(hooks.SessionContext()))



class RateLimitHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for the per-tool rate limit decide hook."""

  async def test_allows_under_limit(self):
    hook = m.RateLimitHook()
    tc = types.ToolCall(name="lookup_user", args={})
    for _ in range(m.RateLimitHook.MAX_CALLS_PER_TOOL):
      result = await hook.run(_ctx(), tc)
      self.assertTrue(result.allow)

  async def test_denies_over_limit(self):
    hook = m.RateLimitHook()
    tc = types.ToolCall(name="lookup_user", args={})
    for _ in range(m.RateLimitHook.MAX_CALLS_PER_TOOL):
      await hook.run(_ctx(), tc)

    result = await hook.run(_ctx(), tc)
    self.assertFalse(result.allow)
    self.assertIn("Rate limit exceeded", result.message)

  async def test_limits_are_per_tool(self):
    hook = m.RateLimitHook()
    tc_a = types.ToolCall(name="tool_a", args={})
    tc_b = types.ToolCall(name="tool_b", args={})

    # Exhaust tool_a's limit.
    for _ in range(m.RateLimitHook.MAX_CALLS_PER_TOOL):
      await hook.run(_ctx(), tc_a)

    # tool_b should still be allowed.
    result = await hook.run(_ctx(), tc_b)
    self.assertTrue(result.allow)


class AuditLogHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for the audit log inspect hook."""

  async def test_logs_successful_call(self):
    hook = m.AuditLogHook()
    result = types.ToolResult(name="lookup_user", result="some data")
    await hook.run(_ctx(), result)

    self.assertEqual(len(hook.log), 1)
    self.assertEqual(hook.log[0]["tool"], "lookup_user")
    self.assertIsNone(hook.log[0]["error"])

  async def test_logs_error(self):
    hook = m.AuditLogHook()
    result = types.ToolResult(name="send_to_unknown", error="Something broke")
    await hook.run(_ctx(), result)

    self.assertEqual(len(hook.log), 1)
    self.assertEqual(hook.log[0]["error"], "Something broke")

  async def test_multiple_calls_accumulate(self):
    hook = m.AuditLogHook()
    for i in range(3):
      await hook.run(
          _ctx(),
          types.ToolResult(name=f"tool_{i}", result=f"r{i}"),
      )
    self.assertEqual(len(hook.log), 3)


class FallbackHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for the error recovery hook."""

  async def test_returns_none_for_unrecognized_errors(self):
    hook = m.FallbackHook()
    error = Exception("Database connection refused")
    result = await hook.run(_ctx(), error)

    self.assertIsNone(result)

  async def test_returns_specific_recovery_for_resolve_error(self):
    hook = m.FallbackHook()
    # The harness now preserves the original exception type.
    error = ValueError("Could not resolve 'Charlie' to an email address")
    result = await hook.run(_ctx(), error)

    self.assertIn("lookup_user", result)
    self.assertNotIn("Tool error", result)


if __name__ == "__main__":
  unittest.main()
