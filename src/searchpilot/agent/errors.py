"""Safe decision failures shared without importing an optional HTTP dependency."""


class DecisionError(Exception):
    """A failed model request must stop the run, rather than trigger paid retries."""

    def __init__(self, code: str) -> None:
        self.code = (
            code
            if code
            in {
                "timeout",
                "authentication_failed",
                "rate_limited",
                "http_error",
                "redirect_rejected",
                "response_too_large",
                "invalid_response",
                "incomplete_response",
                "transport_error",
                "event_sink_error",
                "budget_exhausted",
                "invalid_configuration",
                "invalid_endpoint",
                "invalid_messages",
                "provider_not_open",
                "provider_already_open",
            }
            else "model_error"
        )
        super().__init__(self.code)
