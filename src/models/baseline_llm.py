import os
import json
from gigachat import GigaChat

class BaselineLLM:
    def __init__(self, model_name="GigaChat", credentials=None):
        self.model_name = model_name
        self.credentials = credentials or os.getenv("GIGACHAT_CREDENTIALS")

    def analyze_dialogue(self, prompt_path, dialogue_text):
        with open(prompt_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read()

        print(f"Отправка запроса к {self.model_name}...")
        
        try:
            with GigaChat(credentials=self.credentials, verify_ssl_certs=False) as giga:
                response = giga.chat({
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Текст переговоров:\n\n{dialogue_text}"}
                    ],
                    "temperature": 0.1
                })
                
                result_content = response.choices[0].message.content.strip()
                
                if result_content.startswith("```json"):
                    result_content = result_content[7:]
                if result_content.startswith("```"):
                    result_content = result_content[3:]
                if result_content.endswith("```"):
                    result_content = result_content[:-3]
                
                return json.loads(result_content.strip())
            
        except Exception as e:
            print(f"Ошибка: {e}")
            return None

    def save_result(self, result_data, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(result_data, file, ensure_ascii=False, indent=4)
        print(f"Результат сохранен в {output_path}")
