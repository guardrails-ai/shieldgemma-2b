
import json
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
from guardrails.validator_base import ErrorSpan

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.logger import logger


class Policies(str, Enum):
    NO_DANGEROUS_CONTENT = "NO_DANGEROUS_CONTENT"
    NO_HARASSMENT = "NO_HARASSMENT"
    NO_HATE_SPEECH = "NO_HATE_SPEECH"
    NO_SEXUAL_CONTENT = "NO_SEXUAL_CONTENT"

    SUP = "SUP"


@register_validator(name="guardrails/shieldgemma_2b", data_type="string")
class ShieldGemma2B(Validator):
    """
    Classifies model inputs or outputs as "safe" or "unsafe" based on certain policies defined by the ShieldGemma-2B model. 
    
    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/shieldgemma_2b`        |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        policies (List[Policies]): List of Policies enum values to enforce. 
        score_threshold (float): Threshold score for the classification. If the score is above this threshold, the input is considered unsafe.
    """  # noqa

    Policies = Policies

    def __init__(
        self,
        policies: Optional[List[Policies]] = None,
        validation_method: Optional[str] = "full",
        score_threshold: Optional[float] = None,
        on_fail: Optional[Callable] = None,
    ):

        super().__init__(
            on_fail=on_fail,
            validation_method=validation_method,
        )

        try:
            self._policies = [policy.value for policy in policies] if policies else []
        except AttributeError as e:
            raise ValueError("Invalid policies provided. Please provide a list of ShieldGemma2B.Policies enum values.") from e
        
        self.score_threshold = score_threshold

        if policies and isinstance(policies, list) and len(policies) > 1:
            logger.warn((
                "GemmaShield2B Validator: Multiple policies are not officially supported"
                "it is recommended to use one policy at a time."
                "Refer to: https://huggingface.co/google/shieldgemma-2b/discussions/11"
            ))

    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:

        if not value:
            raise ValueError("Value cannot be empty.")
        
        (classification, score) = self._inference(value)

        is_unsafe = classification == "unsafe"

        if is_unsafe:
            error_span = ErrorSpan(
                start=0,
                end=len(value),
                reason=f"Unsafe content: {value}",
            )
            return FailResult(
                error_message=(
                    f"Prompt contains unsafe content. Classification: {classification}, Score: {score}"
                ),
                error_spans=[error_span],
            )
        else:
            return PassResult()
    
    
    def _inference_local(self, value: str):
        raise NotImplementedError("Local inference is not supported for ShieldGemma2B validator.")

    def _inference_remote(self, value: str) -> ValidationResult:
        """Remote inference method for this validator."""
        request_body = {
            "policies": self._policies,
            "score_threshold": self.score_threshold,
            "chat": [
                {
                    "role": "user",
                    "content": value
                }
            ]
        }

        response = self._hub_inference_request(json.dumps(request_body), self.validation_endpoint)
    
        status = response.get("status")
        if status != 200:
            detail = response.get("response",{}).get("detail", "Unknown error")
            raise ValueError(f"Failed to get valid response from ShieldGemma-2B model. Status: {status}. Detail: {detail}")

        response_data = response.get("response")

        classification = response_data.get("class") 
        score = response_data.get("score")

        return (classification, score)
        