"""Устаревший адаптер baseline, сохранённый для истории миграции.

Новая реализация находится в ``src.judges.llm.judge`` и запускается только через
``python -m src.run_experiment``.
"""

import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Literal

from gigachat import GigaChat
from pydantic import BaseModel, ConfigDict


class CollegiumDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Literal["player_1", "player_2"]


class BaselineLLM:
    COLLEGIUM_ORDER = (
        "collegium_hire",
        "collegium_send",
        "collegium_trust",
    )

    def __init__(self, model_name="GigaChat", credentials=None, temperature=0.1):
        self.model_name = model_name
        self.credentials = credentials or os.getenv("GIGACHAT_CREDENTIALS")
        self.temperature = temperature

    def analyze_dialogue(self, prompt_paths, dialogue_text, input_bout_id):
        if not self.credentials:
            raise ValueError(
                "Не задан GIGACHAT_CREDENTIALS. Добавьте ключ авторизации в файл .env."
            )

        self._validate_prompt_paths(prompt_paths)

        result = {
            "decisions": {collegium: None for collegium in self.COLLEGIUM_ORDER},
            "calls": {},
        }

        for collegium in self.COLLEGIUM_ORDER:
            prompt_path = prompt_paths[collegium]
            call_result = self._analyze_collegium(
                collegium=collegium,
                prompt_path=prompt_path,
                dialogue_text=dialogue_text,
                input_bout_id=input_bout_id,
            )
            result["calls"][collegium] = call_result

            if call_result["status"] == "success":
                result["decisions"][collegium] = call_result["parsed_json"]["winner"]

        return result

    def _validate_prompt_paths(self, prompt_paths):
        expected = set(self.COLLEGIUM_ORDER)
        actual = set(prompt_paths)
        if actual == expected:
            return

        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"отсутствуют: {', '.join(missing)}")
        if unexpected:
            details.append(f"неожиданные: {', '.join(unexpected)}")
        raise ValueError("Некорректный набор промптов (" + "; ".join(details) + ").")

    def _analyze_collegium(
        self,
        collegium,
        prompt_path,
        dialogue_text,
        input_bout_id,
    ):
        raw_response = None
        parsed_json = None
        run_parameters = {
            "model": self.model_name,
            "temperature": self.temperature,
            "input_bout_id": input_bout_id,
            "collegium": collegium,
            "prompt": {
                "path": str(prompt_path),
                "content": None,
            },
            "run_at": datetime.now(timezone.utc).isoformat(),
            "sdk_version": version("gigachat"),
        }

        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                system_prompt = file.read()
            run_parameters["prompt"]["content"] = system_prompt

            print(f"Отправка запроса к {self.model_name}: {collegium}...")

            with GigaChat(credentials=self.credentials, verify_ssl_certs=False) as giga:
                response = giga.chat(
                    {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": f"Текст переговоров:\n\n{dialogue_text}",
                            },
                        ],
                        "temperature": self.temperature,
                        "response_format": {
                            "type": "json_schema",
                            "schema": CollegiumDecision.model_json_schema(),
                            "strict": True,
                        },
                    }
                )

            raw_response = response.choices[0].message.content
            parsed_json = self._parse_response(raw_response)
            validated_result = CollegiumDecision.model_validate(parsed_json).model_dump()
        except Exception as error:
            return {
                "run_parameters": run_parameters,
                "raw_response": raw_response,
                "parsed_json": parsed_json,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }

        return {
            "run_parameters": run_parameters,
            "raw_response": raw_response,
            "parsed_json": validated_result,
            "status": "success",
            "error": None,
        }

    @staticmethod
    def _parse_response(raw_response):
        result_content = raw_response.strip()

        if result_content.startswith("```json"):
            result_content = result_content[7:]
        elif result_content.startswith("```"):
            result_content = result_content[3:]
        if result_content.endswith("```"):
            result_content = result_content[:-3]

        return json.loads(result_content.strip())

    def save_result(self, result_data, output_path):
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result_data, file, ensure_ascii=False, indent=4)
        print(f"Результат сохранен в {output_path}")
