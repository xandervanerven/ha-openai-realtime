"""Tool that lets the model close the conversation gracefully.

Softer sibling of disconnect_tool: instead of tearing down the device
WebSocket, it only suppresses the chime-gated follow-up window at this turn's
natural end (websocket_handler.request_follow_up_chime). The model calls it
when the USER signals they're done ("no thanks", "that's all", "bye"), says
its short goodbye, and the device settles to idle without re-opening the mic.
Next interaction starts with the wake word, which re-enables the chime.
"""
import logging
from typing import Any, Awaitable, Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams
    from app.websocket_handler import WebSocketHandler

logger = logging.getLogger(__name__)


def get_end_conversation_tool_definition() -> Dict[str, Any]:
    """OpenAI-format tool definition (converted to a FunctionSchema for Gemini)."""
    return {
        "type": "function",
        "name": "end_conversation",
        "description": (
            "Close the conversation when the user indicates they are done — e.g. they "
            "answer 'no'/'no thanks'/'nothing else' to your offer of further help, or say "
            "goodbye/thanks-that's-all in any language. Call this, then give ONE short "
            "closing reply (e.g. 'Alright, have a nice evening'). After this tool the "
            "microphone will NOT reopen, so never ask a question after calling it."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def create_end_conversation_tool_handler(
    websocket_handler: "WebSocketHandler",
) -> Callable[["FunctionCallParams"], Awaitable[None]]:
    async def end_conversation_tool_handler(params: "FunctionCallParams") -> None:
        logger.info("👋 end_conversation tool called — suppressing this turn's follow-up chime")
        websocket_handler.suppress_next_followup_chime()
        await params.result_callback(
            "Conversation closed. Give one short goodbye now; the microphone will not reopen."
        )

    return end_conversation_tool_handler
