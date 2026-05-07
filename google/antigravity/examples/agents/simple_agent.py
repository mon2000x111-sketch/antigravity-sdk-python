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

"""Simple agent example using the high-level Agent API.

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. "Creating agent..." appears in the output.
  3. "Chatting with agent..." appears in the output.
  4. The agent produces a response containing "4" (the answer to 2+2).
"""

import asyncio
import logging
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig


async def main():
  logging.basicConfig(level=logging.WARNING)

  print("Creating agent...")
  config = LocalAgentConfig(
      system_instructions="You are a creative novelist assistant.",
  )
  async with Agent(config) as agent:

    prompt = (
        "Simply tell me a 500-word science fiction story about a lonely robot"
        " searching for its creator."
    )
    print(f"\nUser: {prompt}")

    # Instant return — does not block
    response = await agent.chat(prompt)

    # Workflow A: Stream thoughts (reasoning deltas) in real-time!
    print("Agent (Streaming thoughts/reasoning): ")
    print("-------------------------------------------------------")
    async for thought in response.thoughts:
      print(thought, end="", flush=True)
    print("\n-------------------------------------------------------\n")

    # Workflow B: Stream conversational text deltas in real-time!
    print("Agent (Streaming final answer text): ")
    print("-------------------------------------------------------")
    async for token in response:
      print(token, end="", flush=True)
    print("\n-------------------------------------------------------\n")

    print("=== [Stream Exhausted / Cache Resolved] ===")

    # Workflow C: Re-iteration / Lazy accessor replay (Instant cache pull!)
    print("\nPrinting full aggregated text response from in-memory cache:")
    print("-------------------------------------------------------")
    print(await response.text())
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
  asyncio.run(main())
