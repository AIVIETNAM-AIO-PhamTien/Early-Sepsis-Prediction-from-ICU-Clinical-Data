"""LLM providers (free-tier friendly) and template fallback."""

from __future__ import annotations

from abc import ABC, abstractmethod

import requests

from src.narrative.config import get_api_keys
from src.narrative.context import PredictionSummary
from src.narrative.prompts import SYSTEM_PROMPT, build_user_prompt


class NarrativeProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, summary: PredictionSummary, *, language: str) -> str:
        """Return a narrative string."""


class TemplateProvider(NarrativeProvider):
    """Rule-based narrative - no API key, always available."""

    name = "template"

    def generate(self, summary: PredictionSummary, *, language: str) -> str:
        if language == "vi":
            return self._vietnamese(summary)
        return self._english(summary)

    def _vietnamese(self, summary: PredictionSummary) -> str:
        latest_pct = f"{summary.latest_score:.1%}"
        peak_pct = f"{summary.peak_score:.1%}"
        elevated_pct = f"{summary.elevated_pct:.0%}"
        threshold_pct = f"{summary.threshold:.0%}"

        status_vi = {"Low": "thấp", "Watch": "cần theo dõi", "Elevated": "cao"}.get(
            summary.current_status, summary.current_status
        )
        trend_vi = {
            "rising": "đang tăng",
            "falling": "đang giảm",
            "stable": "ổn định",
        }[summary.trend]

        parts = [
            f"Bệnh nhân **{summary.patient_id}** có **{summary.n_hours} giờ** dữ liệu ICU được model chấm điểm.",
            (
                f"Ở **giờ cuối (ICU hour {summary.latest_iculos})**, nguy cơ là **{latest_pct}** "
                f"(mức **{status_vi}**). Ngưỡng cảnh báo là **{threshold_pct}**."
            ),
        ]

        if summary.peak_score >= summary.threshold:
            parts.append(
                f"Trong đợt nằm ICU, điểm cao nhất là **{peak_pct}** tại **giờ {summary.peak_iculos}**."
            )
        else:
            parts.append(
                f"Điểm cao nhất trong đợt nằm viện là **{peak_pct}** - chưa vượt ngưỡng {threshold_pct}."
            )

        if summary.elevated_hours:
            hour_text = ", ".join(str(h) for h in summary.elevated_iculos[:6])
            if len(summary.elevated_iculos) > 6:
                hour_text += ", ..."
            parts.append(
                f"Có **{summary.elevated_hours} giờ** ({elevated_pct} thời gian) vượt ngưỡng "
                f"(ICU hours: {hour_text})."
            )
        else:
            parts.append("Không có giờ nào vượt ngưỡng cảnh báo.")

        parts.append(
            f"Xu hướng gần đây: nguy cơ **{trend_vi}**. "
            "Đây chỉ là ước lượng hỗ trợ quyết định từ model - cần đối chiếu lâm sàng, xét nghiệm và đánh giá bedside."
        )
        return " ".join(parts)

    def _english(self, summary: PredictionSummary) -> str:
        latest_pct = f"{summary.latest_score:.1%}"
        peak_pct = f"{summary.peak_score:.1%}"
        elevated_pct = f"{summary.elevated_pct:.0%}"
        threshold_pct = f"{summary.threshold:.0%}"

        parts = [
            f"Patient **{summary.patient_id}** has **{summary.n_hours} ICU hours** scored by the model.",
            (
                f"At the **latest hour ({summary.latest_iculos})**, risk is **{latest_pct}** "
                f"(**{summary.current_status}**). Alert threshold is **{threshold_pct}**."
            ),
        ]

        if summary.peak_score >= summary.threshold:
            parts.append(
                f"Peak risk during the stay was **{peak_pct}** at **hour {summary.peak_iculos}**."
            )
        else:
            parts.append(f"Peak risk was **{peak_pct}**, below the {threshold_pct} threshold.")

        if summary.elevated_hours:
            hour_text = ", ".join(str(h) for h in summary.elevated_iculos[:6])
            if len(summary.elevated_iculos) > 6:
                hour_text += ", ..."
            parts.append(
                f"**{summary.elevated_hours} hours** ({elevated_pct} of stay) exceeded the threshold "
                f"(ICU hours: {hour_text})."
            )
        else:
            parts.append("No hours exceeded the alert threshold.")

        parts.append(
            f"Recent trend: risk is **{summary.trend}**. "
            "This is a model-based early-warning estimate only - correlate with clinical context."
        )
        return " ".join(parts)


class GeminiProvider(NarrativeProvider):
    """Google Gemini free tier via REST API."""

    name = "gemini"
    default_model = "gemini-2.0-flash"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or self.default_model

    def generate(self, summary: PredictionSummary, *, language: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_user_prompt(summary, language=language)}],
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 512,
            },
        }
        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Gemini response: {data}") from exc


class GroqProvider(NarrativeProvider):
    """Groq free tier (OpenAI-compatible chat API)."""

    name = "groq"
    default_model = "llama-3.1-8b-instant"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or self.default_model

    def generate(self, summary: PredictionSummary, *, language: str) -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.3,
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(summary, language=language),
                    },
                ],
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Groq API error {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Groq response: {data}") from exc


def resolve_provider(
    provider_name: str,
    *,
    gemini_api_key: str | None = None,
    groq_api_key: str | None = None,
) -> NarrativeProvider:
    """Pick a provider; falls back to template when keys are missing."""
    env_gemini, env_groq = get_api_keys()
    gemini_key = gemini_api_key or env_gemini
    groq_key = groq_api_key or env_groq

    if provider_name == "gemini":
        if not gemini_key:
            raise ValueError("Gemini API key not configured.")
        return GeminiProvider(gemini_key)

    if provider_name == "groq":
        if not groq_key:
            raise ValueError("Groq API key not configured.")
        return GroqProvider(groq_key)

    if provider_name == "auto":
        if gemini_key:
            return GeminiProvider(gemini_key)
        if groq_key:
            return GroqProvider(groq_key)
        return TemplateProvider()

    if provider_name == "template":
        return TemplateProvider()

    raise ValueError(f"Unknown provider: {provider_name}")
