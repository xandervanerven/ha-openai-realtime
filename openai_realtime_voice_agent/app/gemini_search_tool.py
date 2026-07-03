"""Gemini-grounded web search handler.

Free alternative to the OpenAI-backed handler in web_search_tool.py: the same
`web_search` function tool (definition reused from there), but the handler runs
a server-side REST call to a non-Live Gemini model with Google Search grounding
enabled. Grounding is rejected on the Live models' free tier (quota 0), but the
Gemini 2/2.5 REST families get 1,500 grounded queries/day free — so the Live
model does the talking and this call does the searching, on the same API key.

Model configurable via GEMINI_SEARCH_MODEL (default gemini-2.5-flash; must be a
Gemini 2.x family model while the Gemini 3 grounding quota is 0).
"""
import logging
from typing import Awaitable, Callable, TYPE_CHECKING

from google import genai
from google.genai import types

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

logger = logging.getLogger(__name__)


def create_gemini_web_search_tool_handler(
    api_key: str, model: str
) -> Callable[["FunctionCallParams"], Awaitable[None]]:
    """Create a web_search handler backed by Gemini + Google Search grounding.

    Returns a short answer via ``params.result_callback`` (which becomes the
    function response the Live model speaks).
    """
    client = genai.Client(api_key=api_key)

    async def web_search_tool_handler(params: "FunctionCallParams") -> None:
        query = (params.arguments or {}).get("query", "").strip()
        logger.info(f"🔎 web_search called: {query!r} (gemini model={model})")

        if not query:
            await params.result_callback("Geen zoekopdracht ontvangen.")
            return

        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=(
                    "Answer in at most 2 short sentences suitable for being read "
                    "aloud, in the same language as the question. Do not include "
                    "URLs, citations, or markdown. Question: " + query
                ),
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            answer = (response.text or "").strip()
            logger.info(f"🔎 web_search answer: {answer[:200]}")
            await params.result_callback(answer or "Ik kon hier online niets over vinden.")
        except Exception as e:
            logger.error(f"❌ web_search failed: {e}", exc_info=True)
            await params.result_callback("Het zoeken op internet lukte even niet.")

    return web_search_tool_handler
