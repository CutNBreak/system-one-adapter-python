"""SDK response types extended with LLM accounting and diagnostics.

Both structs subclass the msgspec SDK types, so a caller serializes them the same way
as any SDK response: `msgspec.to_builtins(response)` or `msgspec.json.encode`. The
`debug` field holds plain JSON-compatible builtins, so no custom serialization is
needed.
"""

from typing import Any

from typesafe_sdk import SystemOneResponse as SDKSystemOneResponse
from typesafe_sdk import Usage as SDKUsage


class Usage(SDKUsage, kw_only=True):
    """Keep final-attempt usage alongside cumulative retry accounting."""

    input_tokens_total: int
    output_tokens_total: int
    n_retries: int
    n_retries_malformed_structure: int
    latency: float


class SystemOneResponse(SDKSystemOneResponse, frozen=True, kw_only=True):
    """SDK answers and typed views with retry and probability diagnostics."""

    usage: Usage
    debug: dict[str, Any]
