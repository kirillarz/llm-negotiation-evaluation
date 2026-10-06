# Методы составления датасета управленческих переговоров

## Общая схема пайплайна


```mermaid
flowchart TD
    A["VK Видео — плейлист ФУБ"] -->|VK Video Downloader| B[" MP4 файлы"]
    B -->|Whisper base| C["TXT файлы"]
    C -->|Qwen 2.5 via Ollama| D["JSON файлы"]
    D -->|Валидация| E["fub_dataset_full.json"]
