import os
import json
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Annotated, Literal

from gigachat import GigaChat
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError


class PlayerScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_1: int = Field(ge=1, le=10, strict=True)
    player_2: int = Field(ge=1, le=10, strict=True)


class NegotiationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Literal["player_1", "player_2"]
    reasoning: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, strict=True)]
    scores: PlayerScores


class BaselineLLM:
    def __init__(self, model_name="GigaChat", credentials=None, temperature=0.1):
        self.model_name = model_name
        self.credentials = credentials or os.getenv("GIGACHAT_CREDENTIALS")
        self.temperature = temperature

    def analyze_dialogue(self, prompt_path, dialogue_text, input_bout_id):
        if not self.credentials:
            raise ValueError(
                "Не задан GIGACHAT_CREDENTIALS. Добавьте ключ авторизации в файл .env."
            )

        with open(prompt_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read()

        run_parameters = {
            "model": self.model_name,
            "temperature": self.temperature,
            "input_bout_id": input_bout_id,
            "prompt": {
                "path": prompt_path,
                "content": system_prompt,
            },
            "run_at": datetime.now(timezone.utc).isoformat(),
            "sdk_version": version("gigachat"),
        }

        print(f"Отправка запроса к {self.model_name}...")

        with GigaChat(credentials=self.credentials, verify_ssl_certs=False) as giga:
            response = giga.chat({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Текст переговоров:\n\n{dialogue_text}"}
                ],
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "schema": NegotiationResult.model_json_schema(),
                    "strict": True,
                },
            })

        raw_response = response.choices[0].message.content
        result_content = raw_response.strip()

        if result_content.startswith("```json"):
            result_content = result_content[7:]
        if result_content.startswith("```"):
            result_content = result_content[3:]
        if result_content.endswith("```"):
            result_content = result_content[:-3]

        parsed_json = None
        try:
            parsed_json = json.loads(result_content.strip())
            validated_result = NegotiationResult.model_validate(parsed_json).model_dump()
        except (json.JSONDecodeError, ValidationError) as error:
            return {
                "run_parameters": run_parameters,
                "raw_response": raw_response,
                "parsed_json": parsed_json,
                "parsing_status": "error",
                "parsing_error": f"{type(error).__name__}: {error}",
                "validated_result": None,
            }

        return {
            "run_parameters": run_parameters,
            "raw_response": raw_response,
            "parsed_json": parsed_json,
            "parsing_status": "success",
            "parsing_error": None,
            "validated_result": validated_result,
        }

    def save_result(self, result_data, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(result_data, file, ensure_ascii=False, indent=4)
        print(f"Результат сохранен в {output_path}")
