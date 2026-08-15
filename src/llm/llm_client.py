from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .output_parser import LLMOutputParseError, LLMOutputParser
from .output_schemas import LLMCallMetadata


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClientConfigurationError(Exception):
    """
    Raised when the LLM client configuration is invalid.
    """


class LLMClientInvocationError(Exception):
    """
    Raised when an LLM invocation fails.

    metadata contains the call information available at the time of failure.
    """

    def __init__(
        self,
        message: str,
        *,
        metadata: LLMCallMetadata | None = None,
    ) -> None:
        super().__init__(message)
        self.metadata = metadata


class LLMClient:
    """
    Workflow-independent LLM client.

    Responsibilities:
    - load environment configuration;
    - initialise or accept an injected chat model;
    - invoke plain-text prompts;
    - invoke prompts with any Pydantic structured-output schema;
    - normalise structured outputs through LLMOutputParser;
    - optionally return latency and token-usage metadata.

    This client deliberately does not know about:
    - Worker A or Worker B;
    - Alice, Bob, Charlie, or Dave;
    - Critic or Coordinator prompt semantics;
    - external retrieval;
    - memory permissions;
    - LangGraph state;
    - any specific benchmark.

    Role-specific behaviour belongs in prompt_templates.py and the workflow.
    """

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 800,
        env_path: str | Path | None = None,
        api_key_env_var: str = "OPENAI_API_KEY",
        validate_api_key: bool = True,
        chat_model: Any | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initialise the client.

        Args:
            model_name:
                Model identifier passed to ChatOpenAI.
            temperature:
                Sampling temperature.
            max_tokens:
                Maximum number of generated tokens.
            env_path:
                Optional path to a .env file.
            api_key_env_var:
                Name of the environment variable containing the API key.
            validate_api_key:
                Whether to validate that the key exists before model creation.
            chat_model:
                Optional injected chat model. Useful for testing and mocks.
                When supplied, ChatOpenAI is not created by this class.
            model_kwargs:
                Additional keyword arguments forwarded to ChatOpenAI.
        """
        model_name = str(model_name).strip()
        api_key_env_var = str(api_key_env_var).strip()

        if not model_name:
            raise LLMClientConfigurationError(
                "model_name cannot be empty."
            )

        if not api_key_env_var:
            raise LLMClientConfigurationError(
                "api_key_env_var cannot be empty."
            )

        if max_tokens <= 0:
            raise LLMClientConfigurationError(
                "max_tokens must be greater than zero."
            )

        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.api_key_env_var = api_key_env_var

        self._structured_model_cache: dict[
            tuple[type[BaseModel], bool],
            Any,
        ] = {}

        if chat_model is not None:
            self.chat_model = chat_model
            return

        self._load_environment(env_path)

        if validate_api_key:
            self._validate_api_key(api_key_env_var)

        chat_model_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        chat_model_kwargs.update(dict(model_kwargs or {}))

        try:
            self.chat_model = ChatOpenAI(**chat_model_kwargs)
        except Exception as error:
            raise LLMClientConfigurationError(
                f"Failed to initialise ChatOpenAI: {error}"
            ) from error

    # ------------------------------------------------------------------
    # Plain-text invocation
    # ------------------------------------------------------------------

    def invoke_text(
        self,
        prompt: str,
    ) -> str:
        """
        Invoke the model and return plain text.
        """
        self._validate_prompt(prompt)

        try:
            response = self.chat_model.invoke(prompt)
            return self._extract_text(response)
        except LLMClientInvocationError:
            raise
        except Exception as error:
            raise LLMClientInvocationError(
                f"Text invocation failed: {error}"
            ) from error

    async def ainvoke_text(
        self,
        prompt: str,
    ) -> str:
        """
        Asynchronously invoke the model and return plain text.
        """
        self._validate_prompt(prompt)

        try:
            response = await self.chat_model.ainvoke(prompt)
            return self._extract_text(response)
        except LLMClientInvocationError:
            raise
        except Exception as error:
            raise LLMClientInvocationError(
                f"Async text invocation failed: {error}"
            ) from error

    def invoke_text_with_metadata(
        self,
        prompt: str,
        *,
        agent_id: str,
        role: str | None = None,
        prompt_preview_chars: int = 500,
    ) -> tuple[str, LLMCallMetadata]:
        """
        Invoke the model and return plain text together with call metadata.
        """
        self._validate_prompt(prompt)
        agent_id = self._validate_identifier(agent_id, "agent_id")

        started_at = perf_counter()

        try:
            response = self.chat_model.invoke(prompt)
            output = self._extract_text(response)
            latency_ms = (perf_counter() - started_at) * 1000.0
            input_tokens, output_tokens = self._extract_token_usage(response)

            metadata = self._build_metadata(
                agent_id=agent_id,
                role=role,
                prompt=prompt,
                raw_output=output,
                success=True,
                error_message=None,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt_preview_chars=prompt_preview_chars,
            )

            return output, metadata
        except Exception as error:
            latency_ms = (perf_counter() - started_at) * 1000.0

            metadata = self._build_metadata(
                agent_id=agent_id,
                role=role,
                prompt=prompt,
                raw_output=None,
                success=False,
                error_message=str(error),
                latency_ms=latency_ms,
                input_tokens=None,
                output_tokens=None,
                prompt_preview_chars=prompt_preview_chars,
            )

            raise LLMClientInvocationError(
                f"Text invocation failed: {error}",
                metadata=metadata,
            ) from error

    # ------------------------------------------------------------------
    # Structured invocation
    # ------------------------------------------------------------------

    def invoke_structured(
        self,
        prompt: str,
        schema_model: type[SchemaT],
        *,
        apply_parser_normalization: bool = True,
    ) -> SchemaT:
        """
        Invoke the model using a Pydantic structured-output schema.

        Raw provider output is requested whenever the installed LangChain
        version supports it. This allows LLMOutputParser to normalize a legacy
        or partially parsed response before final Pydantic validation.
        """
        self._validate_prompt(prompt)
        self._validate_schema_model(schema_model)

        try:
            runnable = self._get_structured_model(
                schema_model,
                include_raw=True,
            )
            response = runnable.invoke(prompt)

            return self._parse_structured_response(
                response=response,
                schema_model=schema_model,
                apply_parser_normalization=apply_parser_normalization,
            )
        except (LLMOutputParseError, LLMClientInvocationError):
            raise
        except Exception as error:
            raise LLMClientInvocationError(
                f"Structured invocation for "
                f"{schema_model.__name__} failed: {error}"
            ) from error

    async def ainvoke_structured(
        self,
        prompt: str,
        schema_model: type[SchemaT],
        *,
        apply_parser_normalization: bool = True,
    ) -> SchemaT:
        """
        Asynchronously invoke the model using a Pydantic schema.
        """
        self._validate_prompt(prompt)
        self._validate_schema_model(schema_model)

        try:
            runnable = self._get_structured_model(
                schema_model,
                include_raw=True,
            )
            response = await runnable.ainvoke(prompt)

            return self._parse_structured_response(
                response=response,
                schema_model=schema_model,
                apply_parser_normalization=apply_parser_normalization,
            )
        except (LLMOutputParseError, LLMClientInvocationError):
            raise
        except Exception as error:
            raise LLMClientInvocationError(
                f"Async structured invocation for "
                f"{schema_model.__name__} failed: {error}"
            ) from error

    def invoke_structured_with_metadata(
        self,
        prompt: str,
        schema_model: type[SchemaT],
        *,
        agent_id: str,
        role: str | None = None,
        apply_parser_normalization: bool = True,
        prompt_preview_chars: int = 500,
    ) -> tuple[SchemaT, LLMCallMetadata]:
        """
        Invoke a structured-output model and return output plus call metadata.
        """
        self._validate_prompt(prompt)
        self._validate_schema_model(schema_model)
        agent_id = self._validate_identifier(agent_id, "agent_id")

        started_at = perf_counter()
        response: Any = None

        try:
            runnable = self._get_structured_model(
                schema_model,
                include_raw=True,
            )
            response = runnable.invoke(prompt)

            output = self._parse_structured_response(
                response=response,
                schema_model=schema_model,
                apply_parser_normalization=apply_parser_normalization,
            )

            latency_ms = (perf_counter() - started_at) * 1000.0
            raw_response = self._raw_response_for_metadata(response)
            usage_response = (
                raw_response
                if raw_response is not None
                else response
            )
            input_tokens, output_tokens = self._extract_token_usage(
                usage_response
            )

            metadata = self._build_metadata(
                agent_id=agent_id,
                role=role,
                prompt=prompt,
                raw_output=self._serialise_raw_output(
                    usage_response
                ),
                success=True,
                error_message=None,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt_preview_chars=prompt_preview_chars,
            )

            return output, metadata

        except Exception as error:
            latency_ms = (perf_counter() - started_at) * 1000.0
            raw_response = self._raw_response_for_metadata(response)
            usage_response = (
                raw_response
                if raw_response is not None
                else response
            )
            input_tokens, output_tokens = self._extract_token_usage(
                usage_response
            )

            metadata = self._build_metadata(
                agent_id=agent_id,
                role=role,
                prompt=prompt,
                raw_output=self._serialise_raw_output(
                    usage_response
                ),
                success=False,
                error_message=str(error),
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt_preview_chars=prompt_preview_chars,
            )

            raise LLMClientInvocationError(
                f"Structured invocation for "
                f"{schema_model.__name__} failed: {error}",
                metadata=metadata,
            ) from error

    async def ainvoke_structured_with_metadata(
        self,
        prompt: str,
        schema_model: type[SchemaT],
        *,
        agent_id: str,
        role: str | None = None,
        apply_parser_normalization: bool = True,
        prompt_preview_chars: int = 500,
    ) -> tuple[SchemaT, LLMCallMetadata]:
        """
        Async version of invoke_structured_with_metadata.
        """
        self._validate_prompt(prompt)
        self._validate_schema_model(schema_model)
        agent_id = self._validate_identifier(agent_id, "agent_id")

        started_at = perf_counter()
        response: Any = None

        try:
            runnable = self._get_structured_model(
                schema_model,
                include_raw=True,
            )
            response = await runnable.ainvoke(prompt)

            output = self._parse_structured_response(
                response=response,
                schema_model=schema_model,
                apply_parser_normalization=apply_parser_normalization,
            )

            latency_ms = (perf_counter() - started_at) * 1000.0
            raw_response = self._raw_response_for_metadata(response)
            usage_response = (
                raw_response
                if raw_response is not None
                else response
            )
            input_tokens, output_tokens = self._extract_token_usage(
                usage_response
            )

            metadata = self._build_metadata(
                agent_id=agent_id,
                role=role,
                prompt=prompt,
                raw_output=self._serialise_raw_output(
                    usage_response
                ),
                success=True,
                error_message=None,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt_preview_chars=prompt_preview_chars,
            )

            return output, metadata

        except Exception as error:
            latency_ms = (perf_counter() - started_at) * 1000.0
            raw_response = self._raw_response_for_metadata(response)
            usage_response = (
                raw_response
                if raw_response is not None
                else response
            )
            input_tokens, output_tokens = self._extract_token_usage(
                usage_response
            )

            metadata = self._build_metadata(
                agent_id=agent_id,
                role=role,
                prompt=prompt,
                raw_output=self._serialise_raw_output(
                    usage_response
                ),
                success=False,
                error_message=str(error),
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt_preview_chars=prompt_preview_chars,
            )

            raise LLMClientInvocationError(
                f"Async structured invocation for "
                f"{schema_model.__name__} failed: {error}",
                metadata=metadata,
            ) from error

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_structured_model_cache(self) -> None:
        """
        Clear cached structured-output runnable instances.
        """
        self._structured_model_cache.clear()

    # ------------------------------------------------------------------
    # Internal structured-output helpers
    # ------------------------------------------------------------------

    def _get_structured_model(
        self,
        schema_model: type[SchemaT],
        *,
        include_raw: bool,
    ) -> Any:
        cache_key = (schema_model, include_raw)

        if cache_key in self._structured_model_cache:
            return self._structured_model_cache[cache_key]

        try:
            runnable = self.chat_model.with_structured_output(
                schema_model,
                include_raw=include_raw,
            )
        except TypeError:
            # Compatibility fallback for LangChain versions that do not expose
            # include_raw. In that case the provider-parsed result is returned
            # directly and raw fallback parsing may be unavailable.
            runnable = self.chat_model.with_structured_output(
                schema_model
            )

        self._structured_model_cache[cache_key] = runnable
        return runnable

    @classmethod
    def _parse_structured_response(
        cls,
        *,
        response: Any,
        schema_model: type[SchemaT],
        apply_parser_normalization: bool,
    ) -> SchemaT:
        """
        Parse a LangChain structured-output response.

        Preferred path:
            provider/LangChain parsed output -> LLMOutputParser -> schema

        Recovery path:
            raw provider content -> LLMOutputParser -> schema

        The recovery path is important for AgentAnswer v2 because provider-side
        Pydantic parsing can fail before the project parser has an opportunity
        to migrate a legacy AgentAnswer v1 response.
        """
        parsed_output, raw_response, provider_error = (
            cls._unpack_structured_response(response)
        )

        errors: list[str] = []

        if parsed_output is not None:
            try:
                return LLMOutputParser.parse_as(
                    parsed_output,
                    schema_model,
                    apply_default_normalizer=(
                        apply_parser_normalization
                    ),
                )
            except LLMOutputParseError as error:
                errors.append(
                    f"parsed-output validation failed: {error}"
                )

        fallback_candidate = cls._structured_fallback_candidate(
            raw_response
        )

        if fallback_candidate is not None:
            try:
                return LLMOutputParser.parse_as(
                    fallback_candidate,
                    schema_model,
                    apply_default_normalizer=(
                        apply_parser_normalization
                    ),
                )
            except LLMOutputParseError as error:
                errors.append(
                    f"raw-output fallback failed: {error}"
                )

        if provider_error is not None:
            errors.insert(
                0,
                f"provider parsing failed: {provider_error}",
            )

        if not errors:
            errors.append(
                "the structured response contained neither a parsed result "
                "nor usable raw content"
            )

        raise LLMOutputParseError(
            f"Failed to parse structured output for "
            f"{schema_model.__name__}: "
            + " | ".join(errors)
        )

    @staticmethod
    def _unpack_structured_response(
        response: Any,
    ) -> tuple[Any, Any, Any]:
        """
        Return (parsed, raw, parsing_error) from an include_raw response.

        A non-wrapper response is treated as a directly parsed output.
        """
        if isinstance(response, Mapping):
            has_structured_wrapper = (
                "parsed" in response
                or "raw" in response
                or "parsing_error" in response
            )

            if has_structured_wrapper:
                return (
                    response.get("parsed"),
                    response.get("raw"),
                    response.get("parsing_error"),
                )

        return response, None, None

    @classmethod
    def _structured_fallback_candidate(
        cls,
        raw_response: Any,
    ) -> Any | None:
        """
        Extract content that LLMOutputParser can parse from a raw response.
        """
        if raw_response is None:
            return None

        content = getattr(raw_response, "content", None)

        if content is None and isinstance(raw_response, Mapping):
            content = raw_response.get(
                "content",
                raw_response.get("text"),
            )

        if isinstance(content, Mapping):
            if "text" not in content and "content" not in content:
                return dict(content)

            content = content.get(
                "text",
                content.get("content"),
            )

        if content is not None:
            text = cls._extract_text_from_content(content)
            if text:
                return text

        if isinstance(raw_response, str):
            text = raw_response.strip()
            return text or None

        if isinstance(raw_response, Mapping):
            return dict(raw_response)

        if isinstance(raw_response, BaseModel):
            return raw_response.model_dump()

        text = str(raw_response).strip()
        return text or None

    @classmethod
    def _raw_response_for_metadata(
        cls,
        response: Any,
    ) -> Any:
        if response is None:
            return None

        _, raw_response, _ = cls._unpack_structured_response(
            response
        )

        return (
            raw_response
            if raw_response is not None
            else response
        )

    # ------------------------------------------------------------------
    # Text and metadata helpers
    # ------------------------------------------------------------------

    @classmethod
    def _extract_text(
        cls,
        response: Any,
    ) -> str:
        if response is None:
            raise LLMClientInvocationError(
                "The model returned no response."
            )

        if isinstance(response, str):
            text = response.strip()
        elif isinstance(response, Mapping):
            text = cls._extract_text_from_content(
                response.get("content", response.get("text"))
            )
        else:
            text = cls._extract_text_from_content(
                getattr(response, "content", response)
            )

        if not text:
            raise LLMClientInvocationError(
                "The model returned an empty text response."
            )

        return text

    @classmethod
    def _extract_text_from_content(
        cls,
        content: Any,
    ) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, Mapping):
            text = content.get("text", content.get("content", ""))
            return cls._extract_text_from_content(text)

        if isinstance(content, (list, tuple)):
            parts: list[str] = []

            for item in content:
                text = cls._extract_text_from_content(item)
                if text:
                    parts.append(text)

            return "\n".join(parts).strip()

        return str(content).strip()

    @staticmethod
    def _extract_token_usage(
        response: Any,
    ) -> tuple[int | None, int | None]:
        if response is None:
            return None, None

        usage_metadata = getattr(response, "usage_metadata", None)

        if isinstance(usage_metadata, Mapping):
            input_tokens = usage_metadata.get(
                "input_tokens",
                usage_metadata.get("prompt_tokens"),
            )
            output_tokens = usage_metadata.get(
                "output_tokens",
                usage_metadata.get("completion_tokens"),
            )

            return (
                LLMClient._safe_non_negative_int(input_tokens),
                LLMClient._safe_non_negative_int(output_tokens),
            )

        response_metadata = getattr(response, "response_metadata", None)

        if isinstance(response_metadata, Mapping):
            token_usage = response_metadata.get(
                "token_usage",
                response_metadata.get("usage"),
            )

            if isinstance(token_usage, Mapping):
                input_tokens = token_usage.get(
                    "prompt_tokens",
                    token_usage.get("input_tokens"),
                )
                output_tokens = token_usage.get(
                    "completion_tokens",
                    token_usage.get("output_tokens"),
                )

                return (
                    LLMClient._safe_non_negative_int(input_tokens),
                    LLMClient._safe_non_negative_int(output_tokens),
                )

        return None, None

    @staticmethod
    def _safe_non_negative_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _serialise_raw_output(
        cls,
        response: Any,
    ) -> str | None:
        if response is None:
            return None

        if isinstance(response, BaseModel):
            try:
                return response.model_dump_json()
            except Exception:
                return str(response).strip() or None

        if isinstance(response, Mapping):
            try:
                return json.dumps(
                    dict(response),
                    ensure_ascii=False,
                    default=str,
                )
            except (TypeError, ValueError):
                return str(response).strip() or None

        if isinstance(response, (list, tuple)):
            try:
                return json.dumps(
                    response,
                    ensure_ascii=False,
                    default=str,
                )
            except (TypeError, ValueError):
                return str(response).strip() or None

        try:
            return cls._extract_text(response)
        except LLMClientInvocationError:
            return str(response).strip() or None

    def _build_metadata(
        self,
        *,
        agent_id: str,
        role: str | None,
        prompt: str,
        raw_output: str | None,
        success: bool,
        error_message: str | None,
        latency_ms: float,
        input_tokens: int | None,
        output_tokens: int | None,
        prompt_preview_chars: int,
    ) -> LLMCallMetadata:
        preview_limit = max(0, int(prompt_preview_chars))
        prompt_preview = (
            prompt[:preview_limit]
            if preview_limit > 0
            else None
        )

        return LLMCallMetadata(
            agent_id=agent_id,
            role=role,
            model_name=self.model_name,
            prompt_preview=prompt_preview,
            raw_output=raw_output,
            success=success,
            error_message=error_message,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ------------------------------------------------------------------
    # Validation and configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _load_environment(
        env_path: str | Path | None,
    ) -> None:
        if env_path is None:
            load_dotenv()
            return

        path = Path(env_path)

        if not path.exists():
            raise LLMClientConfigurationError(
                f"The .env file does not exist: {path}"
            )

        load_dotenv(dotenv_path=path)

    @staticmethod
    def _validate_api_key(
        api_key_env_var: str,
    ) -> None:
        if not os.getenv(api_key_env_var):
            raise LLMClientConfigurationError(
                f"{api_key_env_var} is missing. Put it in the project "
                f"root .env file or configure it in the runtime environment."
            )

    @staticmethod
    def _validate_prompt(
        prompt: str,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

    @staticmethod
    def _validate_schema_model(
        schema_model: type[SchemaT],
    ) -> None:
        if not isinstance(schema_model, type):
            raise TypeError(
                "schema_model must be a Pydantic model class."
            )

        if not issubclass(schema_model, BaseModel):
            raise TypeError(
                "schema_model must inherit from pydantic.BaseModel."
            )

    @staticmethod
    def _validate_identifier(
        value: str,
        field_name: str,
    ) -> str:
        text = str(value).strip()

        if not text:
            raise ValueError(f"{field_name} cannot be empty.")

        return text