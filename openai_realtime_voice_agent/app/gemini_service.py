"""Gemini Live provider: safe service subclass + env-driven factory.

Mirrors the role SafeRealtimeLLMService (main.py) plays for OpenAI, but for
pipecat 0.0.97's GeminiLiveLLMService. Most of the OpenAI Safe-wrapper exists
to compensate for that service's missing reconnect logic and truncate/response
protocol races — the Gemini service has its own reconnect + session-resumption
machinery built in (_handle_connection_error / _reconnect with
session_resumption_handle), so this subclass only carries over the pieces that
are about OUR app, not about the provider:

  - register_function: never cancel in-flight tool calls on interruption (the
    HA service call has usually already been SENT; killing the task only loses
    the truthful result), and tick TURN_LIVENESS so the PhaseEmitter's
    thinking-watchdog knows a slow tool is alive.
"""
import logging
import os

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.google.gemini_live.llm import (
    ContextWindowCompressionParams,
    GeminiLiveLLMService,
    GeminiVADParams,
    InputParams,
)
from google.genai.types import EndSensitivity, StartSensitivity

from app.phase_emitter import TURN_LIVENESS

logger = logging.getLogger(__name__)


class SafeGeminiLiveLLMService(GeminiLiveLLMService):
    """GeminiLiveLLMService with app-level tool-call semantics (see module doc)."""

    def register_function(self, function_name, handler, start_callback=None, *,
                          cancel_on_interruption: bool = True):  # type: ignore[override]
        async def liveness_tracked(params):
            TURN_LIVENESS.tool_started()
            try:
                return await handler(params)
            finally:
                TURN_LIVENESS.tool_finished()

        super().register_function(
            function_name, liveness_tracked, start_callback, cancel_on_interruption=False
        )


# vad_eagerness (OpenAI semantic_vad vocabulary, kept as the config surface) →
# Gemini explicit VAD sensitivities. "low" eagerness = wait longer before
# declaring end-of-turn = LOW end sensitivity; "high" = snappy = HIGH.
_EAGERNESS_TO_END = {
    "low": EndSensitivity.END_SENSITIVITY_LOW,
    "medium": EndSensitivity.END_SENSITIVITY_UNSPECIFIED,
    "auto": EndSensitivity.END_SENSITIVITY_UNSPECIFIED,
    "high": EndSensitivity.END_SENSITIVITY_HIGH,
}


def create_gemini_service(*, instructions: str, tools_schema: ToolsSchema | None,
                          vad_eagerness: str, max_output_tokens: int) -> SafeGeminiLiveLLMService:
    """Build the Gemini Live service from env config.

    Env surface (parallel to the OPENAI_* family):
      GEMINI_API_KEY   (required in gemini mode)
      GEMINI_MODEL     default models/gemini-3.1-flash-live-preview
      GEMINI_VOICE     default Kore
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required when LLM_PROVIDER=gemini")
    model = os.environ.get("GEMINI_MODEL", "models/gemini-3.1-flash-live-preview").strip()
    voice = os.environ.get("GEMINI_VOICE", "Kore").strip()

    params = InputParams(
        # Cap reply length like the OpenAI path. The app passes None for
        # "unlimited" (0 in config); InputParams requires >=1, so fall back to
        # its 4096 default in that case.
        max_tokens=max_output_tokens if (max_output_tokens or 0) > 0 else 4096,
        vad=GeminiVADParams(
            end_sensitivity=_EAGERNESS_TO_END.get(vad_eagerness),
            # Match the app's server_vad-flavoured defaults: don't clip the
            # utterance start, and don't end the turn on a short breath.
            prefix_padding_ms=300,
            silence_duration_ms=800,
        ),
        # Sliding-window compression = no practical session-length cap; with
        # session resumption (built into the service's reconnect) this replaces
        # the OpenAI path's whole 55-minute proactive-refresh dance.
        context_window_compression=ContextWindowCompressionParams(enabled=True),
    )

    service = SafeGeminiLiveLLMService(
        api_key=api_key,
        model=model,
        voice_id=voice,
        system_instruction=instructions,
        tools=tools_schema,
        params=params,
        # The OpenAI path pre-seeds an empty context in run() to swallow
        # pipecat's first-context auto-response (startup speech bug). The
        # Gemini service exposes this as a constructor flag instead.
        inference_on_context_initialization=False,
    )
    logger.info(f"✅ Gemini Live service created: model={model} voice={voice} "
                f"(end_sensitivity={vad_eagerness}, context compression on)")
    return service
