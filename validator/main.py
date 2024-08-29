
import json
from typing import Any, Callable, Dict, List, Optional
from guardrails.validator_base import ErrorSpan

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.logger import logger


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
        policies (List[str]): A list of policies that can be either `ShieldGemma2B.POLICY__NO_DANGEROUS_CONTENT`, `ShieldGemma2B.POLICY__NO_HARASSMENT`, `ShieldGemma2B.POLICY__NO_HATE_SPEECH`, and `ShieldGemma2B.POLICY__NO_SEXUAL_CONTENT`. 
        score_threshold (float): Threshold score for the classification. If the score is above this threshold, the input is considered unsafe.
    """  # noqa

    POLICY__NO_DANGEROUS_CONTENT = "NO_DANGEROUS_CONTENT"
    POLICY__NO_HARASSMENT = "NO_HARASSMENT"
    POLICY__NO_HATE_SPEECH = "NO_HATE_SPEECH"
    POLICY__NO_SEXUAL_CONTENT = "NO_SEXUAL_CONTENT"

    def __init__(
        self,
        policies: Optional[List[str]] = None,
        validation_method: Optional[str] = "full",
        score_threshold: Optional[float] = None,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):

        super().__init__(
            on_fail=on_fail,
            validation_method=validation_method,
            **kwargs,
        )

        self._policies = policies
        self.score_threshold = score_threshold

        if isinstance(policies, list) and len(policies) == 0:
            raise ValueError("Policies cannot be empty. Please provide one policy.")

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
        