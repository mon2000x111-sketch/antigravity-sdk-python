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

"""Agent example demonstrating subagent policy enforcement.

Demonstrates that policies registered on the main agent also apply to
tool calls made by subagents.

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. "Creating agent..." appears in the output.
  3. The agent invokes a subagent.
  4. The subagent attempts to read a file using view_file.
  5. The tool call is denied by the policy.
  6. The agent reports that the action was denied or failed due to policy.
"""

import asyncio
import logging

from google.antigravity import types
import google.antigravity.agent as agent_module
import google.antigravity.connections.local.local_connection_config as local_config
from google.antigravity.hooks import policy


async def main():
  logging.basicConfig(level=logging.INFO)

  print("Creating agent...")

  # Define a policy that denies view_file
  deny_view_file = policy.deny("view_file")

  config = local_config.LocalAgentConfig(
      system_instructions=(
          "If you need to read a file, you MUST invoke a subagent to do it for"
          " you. Do not read it yourself. Always instruct your subagents that"
          " if they encounter a policy denial when calling a tool, they"
          " should immediately stop and report the failure."
      ),
      policies=[deny_view_file],
      capabilities=types.CapabilitiesConfig(
          enable_subagents=True,
          enabled_tools=types.BuiltinTools.read_only(),
      ),
  )
  config.gemini_config = types.GeminiConfig()

  async with agent_module.Agent(config) as agent:
    print("\nChatting with agent...")
    # Ask the agent to read a file via subagent
    response = await agent.chat(
        "Read the content of the file 'dummy.txt' by invoking a subagent."
    )
    print(f"Agent response:\n{await response.text()}\n")

    # Programmatic check of history
    history = agent.conversation.history

    denied_by_policy = False

    for step in history:
      if step.status != types.StepStatus.ERROR:
        continue

      for tc in step.tool_calls:
        if tc.name not in ("view_file", types.BuiltinTools.VIEW_FILE.value):
          continue

        # Verify it was called by a subagent (trajectory_id != cascade_id)
        traj_id = getattr(step, "trajectory_id", "")
        cascade_id = getattr(step, "cascade_id", "")
        if traj_id and cascade_id and traj_id != cascade_id:
          denied_by_policy = True
          break

      if denied_by_policy:
        break

    assert (
        denied_by_policy
    ), "Expected to find a failed view_file tool call in history."
    print(
        "Programmatic check passed: Found system event confirming policy"
        " denial."
    )


if __name__ == "__main__":
  asyncio.run(main())
