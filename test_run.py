"""Устаревший baseline-прототип.

Не использовать для нового сбора голосов. Поддерживаемая точка входа:
``python -m src.run_experiment``.
"""

import json
from dotenv import load_dotenv
from src.models.baseline_llm import BaselineLLM

# Загружаем переменные из файла .env
load_dotenv()

DATASET_PATH = "data/dataset/fub_video_transcript.json"
OUTPUT_PATH = "results/pilot/test_baseline_result.json"
PROMPT_PATHS = {
    "collegium_hire": "prompts/pilot/hiring.txt",
    "collegium_send": "prompts/pilot/sending-to-negotiations.txt",
    "collegium_trust": "prompts/pilot/trustee-of-property.txt",
}


def format_dialogue_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dialogue_text = f"Исходная ситуация:\n{data['situation']}\n\nПереговорные раунды:\n"
    for round_data in data['rounds']:
        dialogue_text += f"\n--- Раунд {round_data['round_id']} ---\n"
        for turn in round_data['turns']:
            role = round_data['roles'][turn['speaker']]
            dialogue_text += f"[{turn['speaker']}] {role}: {turn['text']}\n"
            
    return str(data['id']), dialogue_text

def main():
    llm = BaselineLLM(model_name="GigaChat-2-Pro")

    print("Чтение файла датасета...")
    input_bout_id, dialogue_text = format_dialogue_from_json(DATASET_PATH)
    print("Текст сформирован. Отправляем в модель...")

    result = llm.analyze_dialogue(PROMPT_PATHS, dialogue_text, input_bout_id)

    llm.save_result(result, OUTPUT_PATH)
    failed_collegia = [
        collegium
        for collegium, call in result["calls"].items()
        if call["status"] != "success"
    ]
    if failed_collegia:
        raise RuntimeError(
            "Не удалось получить решения коллегий: "
            f"{', '.join(failed_collegia)}. "
            f"Диагностика сохранена в {OUTPUT_PATH}"
        )

    print("\nУспех! Получены решения трех коллегий:")
    print(json.dumps(result["decisions"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
