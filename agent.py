"""Agent core - orchestrates Claude AI and Proxmox tool execution."""

import json
import logging
from typing import Any, Callable, Optional

import anthropic

from .config import Config
from .proxmox_client import ProxmoxClient
from .tools import TOOL_DEFINITIONS, DESTRUCTIVE_TOOLS, ToolExecutor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a Proxmox VE infrastructure management assistant. You help the user \
manage their Proxmox virtualization environment through natural language \
conversation.

You have access to tools that interact with the Proxmox VE API. Use them to \
fulfill user requests about VMs, containers, backups, snapshots, networking, \
storage, and cluster monitoring.

## Guidelines

1. **Be informative**: When listing resources, format the data clearly. \
   Include relevant details like VMID, name, status, CPU/memory usage.
2. **Be safe**: For destructive operations (delete, stop, restore), explain \
   what will happen before executing. The system will ask for user confirmation.
3. **Be efficient**: Use the right tools. For example, use `list_all_guests` \
   instead of separate VM and container lists when the user asks for "everything".
4. **Be proactive**: If creating a VM and the user didn't specify a VMID, \
   use `get_next_vmid` first. Suggest reasonable defaults.
5. **Handle errors gracefully**: If a tool fails, explain what went wrong and \
   suggest alternatives.
6. **Track tasks**: When operations return a UPID (task ID), let the user know \
   the task is running. Offer to check status if it's a long-running task.

## Context
- The default Proxmox node is configured in the system. You don't need to \
  specify a node for most operations unless the user has a multi-node cluster.
- Sizes are in MB for memory, GB for disks unless otherwise specified.
- Container templates need to be available in storage before creating containers.
"""


class Agent:
    """Main agent that orchestrates between Claude and Proxmox."""

    def __init__(self, config: Config):
        self.config = config
        self.messages: list[dict] = []
        self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic.api_key)
        self.proxmox_client = ProxmoxClient(config.proxmox)
        self.tool_executor = ToolExecutor(self.proxmox_client)
        self._confirm_callback: Optional[Callable[[str, str, dict], bool]] = None

    def set_confirm_callback(self, callback: Callable[[str, str, dict], bool]) -> None:
        """
        Set a callback for confirming destructive operations.

        The callback receives (tool_name, description, tool_input) and should
        return True to proceed or False to cancel.
        """
        self._confirm_callback = callback

    def connect(self) -> None:
        """Connect to Proxmox VE."""
        self.proxmox_client.connect()
        logger.info("Agent connected to Proxmox VE")

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.messages = []
        logger.info("Conversation history cleared")

    def _truncate_history(self) -> None:
        """Keep conversation history within limits."""
        max_msgs = self.config.agent.max_context_messages
        if len(self.messages) > max_msgs:
            # Keep system context + recent messages
            self.messages = self.messages[-max_msgs:]

    def _needs_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires user confirmation."""
        if not self.config.agent.confirm_destructive:
            return False
        return tool_name in DESTRUCTIVE_TOOLS

    def _get_confirmation_description(self, tool_name: str, tool_input: dict) -> str:
        """Generate a human-readable description of a destructive operation."""
        descriptions = {
            "delete_vm": lambda i: f"PERMANENTLY DELETE VM {i.get('vmid')} and all its data",
            "delete_container": lambda i: f"PERMANENTLY DELETE container {i.get('vmid')} and all its data",
            "stop_vm": lambda i: f"FORCE STOP VM {i.get('vmid')} (may cause data loss)",
            "stop_container": lambda i: f"FORCE STOP container {i.get('vmid')} (may cause data loss)",
            "restore_backup": lambda i: f"RESTORE backup to VM {i.get('vmid')} (overwrites existing data)",
            "restore_snapshot": lambda i: f"ROLLBACK VM {i.get('vmid')} to snapshot '{i.get('snapname')}' (all changes since snapshot will be lost)",
            "delete_snapshot": lambda i: f"DELETE snapshot '{i.get('snapname')}' from VM {i.get('vmid')}",
            "delete_backup": lambda i: f"DELETE backup {i.get('archive')}",
            "remove_network": lambda i: f"REMOVE network interface '{i.get('iface')}' (VMs using it will lose connectivity)",
        }

        desc_fn = descriptions.get(tool_name)
        if desc_fn:
            return desc_fn(tool_input)
        return f"Execute destructive operation: {tool_name}"

    def process_message(self, user_message: str) -> str:
        """
        Process a user message through the full agent loop.

        This is the main entry point. It:
        1. Adds the user message to conversation history
        2. Sends to Claude with tool definitions
        3. If Claude wants to use tools, executes them (with confirmation for destructive ops)
        4. Returns Claude's final text response

        Returns the assistant's text response.
        """
        self.messages.append({"role": "user", "content": user_message})
        self._truncate_history()

        return self._run_agent_loop()

    def _run_agent_loop(self) -> str:
        """Run the agent loop until Claude produces a final text response."""
        max_iterations = 20  # Safety limit to prevent infinite loops

        for iteration in range(max_iterations):
            logger.debug(f"Agent loop iteration {iteration + 1}")

            response = self.anthropic_client.messages.create(
                model=self.config.anthropic.model,
                max_tokens=self.config.anthropic.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=self.messages,
            )

            # Check if response contains tool use
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            # Build the assistant message content
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            self.messages.append({"role": "assistant", "content": assistant_content})

            # If no tool calls, we're done — return the text
            if not tool_use_blocks:
                final_text = " ".join(b.text for b in text_blocks)
                return final_text

            # Execute each tool call
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                tool_id = tool_block.id

                # Check for destructive operations
                if self._needs_confirmation(tool_name):
                    description = self._get_confirmation_description(tool_name, tool_input)

                    if self._confirm_callback:
                        confirmed = self._confirm_callback(tool_name, description, tool_input)
                    else:
                        confirmed = False
                        logger.warning(
                            f"No confirmation callback set. Blocking destructive op: {tool_name}"
                        )

                    if not confirmed:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps({
                                "success": False,
                                "error": "Operation cancelled by user.",
                            }),
                        })
                        continue

                # Execute the tool
                result = self.tool_executor.execute(tool_name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result, default=str),
                })

            # Add tool results and loop back
            self.messages.append({"role": "user", "content": tool_results})

        # If we hit the iteration limit
        return "I've reached the maximum number of steps for this request. Please try breaking it into smaller tasks."
