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
  - _connection_task_handler: adds the GoAway handling pipecat 0.0.97 lacks
    (graceful reconnect on Google's pre-termination notice, instead of letting
    the socket die and reconnecting through the error path).
"""
import logging
import os
import time

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

    async def _handle_connection_error(self, error: Exception) -> bool:  # type: ignore[override]
        """Don't let routine idle-session drops accumulate into a fatal death.

        Google aborts quiet Live sessions every ~2.5 min (ws close 1008 "The
        operation was aborted"); the built-in reconnect restores them with the
        session-resumption handle. But pipecat's consecutive-failure counter
        only resets on session ACTIVITY, which an idle session never generates
        — so the third quiet abort marks the service fatally dead (observed
        live 2026-07-03: dead after ~7.5 min of silence). An abort while
        nothing is in flight is routine cycling, not degradation: reconnect
        without counting it. Mid-turn failures still count toward fatal.
        """
        if not getattr(self, "_bot_is_responding", False) and not getattr(self, "_user_is_speaking", False):
            self._consecutive_failures = 0
            logger.info(f"🔁 idle-session drop ({error}) — reconnecting, not counted toward fatal")
            return True
        return await super()._handle_connection_error(error)

    async def _connection_task_handler(self, config):  # type: ignore[override]
        """Copy of the pinned-0.0.97 receive loop + GoAway handling.

        Google sends a GoAway message (with time_left) before terminating a
        connection; the base loop's if/elif chain silently drops it, so the
        socket later dies hard (ws 1008) and recovery runs through the ERROR
        path. Here GoAway instead triggers a graceful in-place reconnect — at
        once when the session is quiet, else at the next turn boundary — using
        the same _reconnect()+return pattern (and the retained session
        resumption handle) as the error path, without touching the failure
        counter. If a mid-turn GoAway kills the socket before the turn ends,
        the error path still catches it exactly as before.
        """
        async with self._client.aio.live.connect(model=self._model_name, config=config) as session:
            logger.info("Connected to Gemini service")
            self._connection_start_time = time.time()
            await self._handle_session_ready(session)

            goaway_pending = False
            while True:
                try:
                    turn = self._session.receive()
                    async for message in turn:
                        self._check_and_reset_failure_counter()

                        if getattr(message, "go_away", None):
                            goaway_pending = True
                            time_left = getattr(message.go_away, "time_left", None)
                            logger.info(f"👋 GoAway from server (time_left={time_left}) — "
                                        "will reconnect gracefully at the next quiet moment")
                        elif message.server_content and message.server_content.model_turn:
                            await self._handle_msg_model_turn(message)
                        elif (
                            message.server_content
                            and message.server_content.turn_complete
                            and message.usage_metadata
                        ):
                            await self._handle_msg_turn_complete(message)
                            await self._handle_msg_usage_metadata(message)
                        elif message.server_content and message.server_content.input_transcription:
                            await self._handle_msg_input_transcription(message)
                        elif message.server_content and message.server_content.output_transcription:
                            await self._handle_msg_output_transcription(message)
                        elif message.server_content and message.server_content.grounding_metadata:
                            await self._handle_msg_grounding_metadata(message)
                        elif message.tool_call:
                            await self._handle_msg_tool_call(message)
                        elif message.session_resumption_update:
                            self._handle_msg_resumption_update(message)

                        if goaway_pending and not self._bot_is_responding and not self._user_is_speaking:
                            break  # quiet moment mid-stream — reconnect below
                    if goaway_pending and not self._bot_is_responding and not self._user_is_speaking:
                        logger.info("🔁 graceful reconnect after GoAway (session resumption keeps context)")
                        await self._reconnect()
                        return
                except Exception as e:
                    if not self._disconnecting:
                        should_reconnect = await self._handle_connection_error(e)
                        if should_reconnect:
                            await self._reconnect()
                            return
                    break

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
