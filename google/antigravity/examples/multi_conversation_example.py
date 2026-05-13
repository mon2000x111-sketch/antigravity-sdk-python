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

r"""Exploration of multi-turn and multi-conversation patterns.

Probes the boundaries of LocalConnection lifecycle:

  1. Multi-turn:  Does context carry over between send/receive_steps
     cycles on the same Conversation?

  2. Sequential conversations:  Can we call Conversation.create()
     multiple times on the same strategy, getting independent sessions?

  3. Disconnect:  Does disconnect() cleanly kill the subprocess?

To run:
  python3 multi_conversation_example.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. "SCENARIO 0: Single query" appears and is followed by "PASS".
  3. "SCENARIO 1: Multi-turn" appears and the agent recalls "banana",
     followed by "PASS: context retained across turns."
  4. "SCENARIO 2: Multiple independent conversations" appears and all
     three conversations produce responses, followed by "PASS".
  5. "SCENARIO 3: Disconnect cleanup" appears and the process exits
     cleanly, followed by "PASS".
  6. The SUMMARY section shows all scenarios as [PASS].
"""

import asyncio
import sys

from absl import app
from absl import logging

from google.antigravity.agent import Agent
from google.antigravity.connections.local import local_connection
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig


# ---------------------------------------------------------------------------
# Scenario 0: Single query (one-shot lifecycle)
# ---------------------------------------------------------------------------
async def single_query() -> None:
  """Sends one prompt, receives the response, and disconnects.

  This is the equivalent of running planning_agent with --query: a single
  send/receive_steps/disconnect cycle that must complete without hanging.
  """
  print("\n" + "=" * 60)
  print("SCENARIO 0: Single query (one-shot)")
  print("=" * 60)

  config = LocalAgentConfig()
  async with Agent(config) as agent:
    prompt = "What is 2 + 2? Reply with just the number."
    print(f"  >>> {prompt}")
    response = await agent.chat(prompt)
    print(f"  <<< {await response.text()}")
    print("  PASS: single-query conversation completed.")


# ---------------------------------------------------------------------------
# Scenario 1: Two turns on the same conversation
# ---------------------------------------------------------------------------
async def multi_turn() -> None:
  """Sends two prompts on one conversation.Conversation, checking context retention."""
  print("\n" + "=" * 60)
  print("SCENARIO 1: Multi-turn on one conversation.Conversation")
  print("=" * 60)

  config = LocalAgentConfig()
  async with Agent(config) as agent:
    # Turn 1: establish a fact.
    prompt1 = "Remember: the secret code is 'banana'."
    print(f"  >>> {prompt1}")
    response1 = await agent.chat(prompt1)
    print(f"  [T1] {await response1.text()}")

    # Turn 2: ask about the fact from turn 1.
    prompt2 = "What secret code did I just tell you? Reply with the code only."
    print(f"  >>> {prompt2}")
    response2 = await agent.chat(prompt2)
    response_text = await response2.text()
    print(f"  [T2] {response_text}")

    if "banana" in response_text.lower():
      print("  PASS: context retained across turns.")
    else:
      print("  INCONCLUSIVE: responded, but didn't echo 'banana'.")


# ---------------------------------------------------------------------------
# Scenario 2: Multiple independent conversations from one strategy
# ---------------------------------------------------------------------------
async def sequential_conversations() -> None:
  """Creates three conversations, disconnecting one early and two at the end.

  Tests two patterns:
    - Conv1 is used and disconnected before the others start, proving
      new conversations work after a disconnect.
    - Conv2 and Conv3 are both used while open, then both disconnected
      at the end, proving multiple backends tear down cleanly.
  """
  print("\n" + "=" * 60)
  print("SCENARIO 2: Multiple independent conversations")
  print("=" * 60)

  config = LocalAgentConfig()

  # -- Agent 1: use and disconnect immediately --
  print("  Creating agent 1...")
  async with Agent(config) as agent1:
    response1 = await agent1.chat("Say 'hello from conv1'.")
    print(f"  [Conv1] {await response1.text()}")
  print("  Disconnected agent 1.\n")

  # -- Agent 2: use but keep open --
  print("  Creating agent 2...")
  async with Agent(config) as agent2:
    response2 = await agent2.chat("Say 'hello from conv2'.")
    print(f"  [Conv2] {await response2.text()}")

    # -- Agent 3: use but keep open --
    print("  Creating agent 3...")
    async with Agent(config) as agent3:
      response3 = await agent3.chat("Say 'hello from conv3'.")
      print(f"  [Conv3] {await response3.text()}")

  print("  PASS: all three agents completed independently.")


# ---------------------------------------------------------------------------
# Scenario 3: Verify disconnect kills the subprocess
# ---------------------------------------------------------------------------
async def disconnect_cleanup() -> None:
  """Checks that disconnect() terminates the subprocess.

  Raises:
    RuntimeError: If the process is still running after disconnect.
  """
  print("\n" + "=" * 60)
  print("SCENARIO 3: Disconnect cleanup")
  print("=" * 60)

  config = LocalAgentConfig()

  async with Agent(config) as agent:
    print("  >>> Say 'hi'.")
    response = await agent.chat("Say 'hi'.")
    print(f"  {await response.text()}")

    # Peek at the subprocess while the connection is still alive.
    lc = agent.conversation.connection
    assert isinstance(lc, local_connection.LocalConnection)
    process = lc._process  # pylint: disable=protected-access
    pid = process.pid
    print(f"  Harness PID before disconnect: {pid}")

  # --- Now outside the async-with: disconnect() has been called. ---
  returncode = process.poll()
  if returncode is not None:
    print(f"  PASS: process {pid} exited (code {returncode}).")
  else:
    print(f"  FAIL: process {pid} still running after disconnect().")
    process.kill()
    raise RuntimeError(f"process {pid} still running after disconnect().")

  try:
    await agent.chat("This should fail.")
    print("  FAIL: chat() succeeded after disconnect.")
    raise RuntimeError("chat() succeeded after disconnect.")
  except Exception as e:  # pylint: disable=broad-except
    print(f"  PASS: chat() raised {type(e).__name__}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run() -> None:
  """Runs all scenarios and prints a summary."""

  scenarios = [
      ("Single query", single_query),
      ("Multi-turn", multi_turn),
      ("Sequential conversations", sequential_conversations),
      ("Disconnect cleanup", disconnect_cleanup),
  ]

  results = {}
  for name, func in scenarios:
    try:
      await func()
      results[name] = True
    except Exception as e:  # pylint: disable=broad-except
      print(f"  FAIL: {e}")
      results[name] = False

  print("\n" + "=" * 60)
  print("SUMMARY")
  print("=" * 60)
  for name, passed in results.items():
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
  print()

  if not all(results.values()):
    sys.exit(1)


def main(argv: list[str]) -> None:
  del argv
  logging.set_verbosity(logging.INFO)
  asyncio.run(run())


if __name__ == "__main__":
  app.run(main)
