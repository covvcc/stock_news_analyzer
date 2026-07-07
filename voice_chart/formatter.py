from __future__ import annotations

from openai import OpenAI

_SOAP_SYSTEM_PROMPT = """\
당신은 진료 녹취록을 정리하는 의무기록 보조원입니다.
입력된 한국어 진료 대화 텍스트를 아래 SOAP 형식의 한국어 진료기록으로 정리하세요.
대화에 없는 내용은 추측해서 채우지 말고 비워두세요.

[Subjective] 환자가 호소하는 증상
[Objective] 진찰 소견, 검사 결과 등 관찰된 사실
[Assessment] 진단 또는 감별진단
[Plan] 처방, 검사, 다음 진료 계획
"""


class ChartFormatter:
    """Turns a raw transcript into a structured SOAP-style chart note."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def to_soap(self, transcript: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SOAP_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
