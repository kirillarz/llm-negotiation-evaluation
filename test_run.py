import os
import json
from dotenv import load_dotenv
from src.models.baseline_llm import BaselineLLM

# Загружаем переменные из файла .env
load_dotenv()

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
    
    dataset_path = "data/pilot_dataset/fub_video_transcript.json"
    prompt_path = "prompts/pilot/baseline_prompt.txt"
    output_path = "results/pilot/test_baseline_result.json"
    
    print("Чтение файла датасета...")
    input_bout_id, dialogue_text = format_dialogue_from_json(dataset_path)
    print("Текст сформирован. Отправляем в модель...")
    
    result = llm.analyze_dialogue(prompt_path, dialogue_text, input_bout_id)
    
    llm.save_result(result, output_path)
    if result["parsing_status"] != "success":
        raise RuntimeError(
            f"Ответ модели не прошел проверку: {result['parsing_error']}. "
            f"Диагностика сохранена в {output_path}"
        )

    print("\nУспех! Модель ответила:")
    print(json.dumps(result["validated_result"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
