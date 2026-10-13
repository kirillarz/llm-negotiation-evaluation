import json
import glob
from pydantic import BaseModel, ValidationError, model_validator
from typing import List

class Roles(BaseModel):
    player_1: str
    player_2: str

class Turn(BaseModel):
    turn_id: int
    speaker: str
    text: str

class Round(BaseModel):
    round_id: int
    roles: Roles
    turns: List[Turn]

class Vote(BaseModel):
    judge_id: int
    collegia: str
    decision: str
    comment: str

class Score(BaseModel):
    player_1: int
    player_2: int

class ResultDef(BaseModel):
    score: Score

class MetadataDef(BaseModel):
    date: str
    source_file: str

class TranscriptSchema(BaseModel):
    id: str
    situation: str
    rounds: List[Round]
    human_votes: List[Vote]
    result: ResultDef
    metadata: MetadataDef
    schema_version: str

    @model_validator(mode='after')
    def check_votes_and_score(self):
        votes_p1 = sum(1 for vote in self.human_votes if vote.decision == "player_1")
        votes_p2 = sum(1 for vote in self.human_votes if vote.decision == "player_2")
        
        if self.result.score.player_1 != votes_p1 or self.result.score.player_2 != votes_p2:
            raise ValueError(
                f"ОШИБКА ПОДСЧЕТА: Фактические голоса (player_1: {votes_p1}, player_2: {votes_p2}) "
                f"не совпадают с итоговым счетом в result.score (player_1: {self.result.score.player_1}, player_2: {self.result.score.player_2})."
            )
        return self

def test_all_transcriptions(dataset_folder="dataset/*.json"):
    files = glob.glob(dataset_folder)
    if not files:
        print("Файлы транскрипций не найдены!")
        return

    all_passed = True
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            TranscriptSchema(**data)
            print(f"[OK] Файл {file_path} прошел проверку структуры и подсчета голосов.")
        except ValidationError as e:
            all_passed = False
            print(f"[ОШИБКА] В файле {file_path}:")
            for error in e.errors():
                print(f"    - Поле: {error.get('loc')} -> {error.get('msg')}")
        except Exception as e:
            all_passed = False
            print(f"[ОШИБКА] Не удалось прочитать {file_path}: {e}")

    if all_passed:
        print("\nВсе тесты успешно пройдены!")

if __name__ == "__main__":
    test_all_transcriptions()