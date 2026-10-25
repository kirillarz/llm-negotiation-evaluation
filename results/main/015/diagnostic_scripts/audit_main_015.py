"""Independently verify provider choices and main-run summary metrics."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src/config.yaml").is_file()),
            Path(r"C:\Users\Денис\Desktop\Decision_experiment"))
OUT = ROOT / "results/main/015"
COLLEGIA = ("hiring", "negotiation", "ownership")


def swap(label):
    assert label in ("player_1", "player_2")
    return "player_2" if label == "player_1" else "player_1"


def audit_payloads():
    sys.path.insert(0, str(ROOT))
    from src.experiment.config import load_config
    from src.experiment.data_loader import load_dataset, swap_players
    from src.experiment.prompts import load_prompts
    from src.judges.decision.judge import DecisionJudge
    from src.judges.llm.judge import LLMJudge

    cfg = load_config(ROOT / "src/config.yaml")
    prompts = load_prompts(cfg.prompts)
    judges = (LLMJudge(cfg.models.llm, prompts), DecisionJudge(cfg.models.decision, prompts))
    pairs, turns = [], 0
    for variants in load_dataset(cfg, "main"):
        a, b = variants.original, variants.swapped
        assert swap_players(b) == a
        assert a.situation == b.situation
        assert set(a.model_dump()) == {"situation", "rounds"}
        for x, y in zip(a.rounds, b.rounds, strict=True):
            assert x.round_id == y.round_id
            assert x.roles.player_1 == y.roles.player_2 and x.roles.player_2 == y.roles.player_1
            for t, s in zip(x.turns, y.turns, strict=True):
                assert t.turn_id == s.turn_id and t.text == s.text and t.speaker == swap(s.speaker)
                turns += 1
        for judge in judges:
            for collegium in COLLEGIA:
                original = judge._request_payload(a, collegium)
                swapped = judge._request_payload(b, collegium)
                if isinstance(judge, LLMJudge):
                    x, y = [json.JSONDecoder().raw_decode(p["messages"][1]["content"].split("\n", 1)[1])[0]
                            for p in (original, swapped)]
                    assert original["messages"][0] == swapped["messages"][0]
                else:
                    x, y = original["state"], swapped["state"]
                    assert original["questions"] == swapped["questions"]
                assert x == a.model_dump(mode="json") and y == b.model_dump(mode="json")
                digest = lambda p: hashlib.sha256(json.dumps(p, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                one, two = digest(original), digest(swapped)
                assert one != two
                pairs.append({"bout_id": variants.bout_id, "model": judge.model_alias,
                              "collegium": collegium, "original_sha256": one, "swapped_sha256": two})
    assert len(pairs) == 150
    (OUT / "request_payload_audit.json").write_text(json.dumps({
        "method": "Offline reconstruction using unchanged code and inputs; no additional API requests",
        "pairs_checked": len(pairs), "original_turns_checked": turns,
        "all_rounds_and_turns_preserved": True, "canonicalization_absent": True,
        "human_votes_absent_from_prepared_bouts": True, "pairs": pairs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    snapshot = json.loads((OUT / "evaluation_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["capture_stage"] == "Before the first API call"
    assert metadata["run_id"] == "015"
    assert len(metadata["planned_tasks"]) == 900
    assert {m["api_model_id"] for m in metadata["models"]} == {"clef", "qwen3.8-flash"}
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == item["sha256"]
               for name, item in snapshot["files"].items())
    audit_payloads()
    history = [json.loads(line) for line in (OUT / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    latest = {}
    for row in history:
        task = row["task"]
        key = tuple(task[k] for k in ("model_alias", "bout_id", "collegium", "variant", "repeat_index"))
        latest[key] = row
    planned = {tuple(task[k] for k in ("model_alias", "bout_id", "collegium", "variant", "repeat_index"))
               for task in metadata["planned_tasks"]}
    assert set(latest) == planned and len(latest) == 900
    assert all(row["status"] == "success" for row in latest.values())
    grouped = defaultdict(lambda: defaultdict(dict))
    for (model, bout, collegium, variant, repeat), row in latest.items():
        raw = row["response"]["raw_response"]
        native = (json.loads(raw["choices"][0]["message"]["content"])["choice"]
                  if model.startswith("qwen") else raw["answers"]["vote"]["choice"])
        assert native == row["response"]["winner"]
        assert "_adapter" not in raw
        if model.startswith("qwen"):
            assert row["effective_parameters"]["reasoning"]["effort"] == "none"
            assert (row["response"]["usage"] or {})["completion_tokens_details"]["reasoning_tokens"] == 0
        grouped[(model, bout, collegium)][variant][repeat] = native if variant == "original" else swap(native)
    with (OUT / "metrics/per_bout_collegium.csv").open(encoding="utf-8", newline="") as stream:
        standard = {(r["model_alias"], r["bout_id"], r["collegium"]): r for r in csv.DictReader(stream)}
    details = []
    for (model, bout, collegium), variants in sorted(grouped.items()):
        original, swapped = [[variants[v][i] for i in (1, 2, 3)] for v in ("original", "swapped")]
        a, b = [Counter(v).most_common(1)[0][0] for v in (original, swapped)]
        official = standard[(model, bout, collegium)]
        assert (official["position_winner_changed"] == "True") == (a != b)
        assert official["original_majority"] == a
        details.append({
            "model": model, "bout_id": bout, "collegium": collegium,
            "original": original, "swapped_mapped": swapped,
            "original_majority": a, "swapped_majority": b, "majority_changed": a != b,
            "human_majority": official["human_majority"],
            "human_agreement_original": a == official["human_majority"],
            "original_unanimous": len(set(original)) == 1,
            "swapped_unanimous": len(set(swapped)) == 1,
            "equal_repeat_pairs": sum(x == y for values in (original, swapped) for x, y in combinations(values, 2)),
            "cross_disagree_pairs": sum(x != y for x, y in product(original, swapped)),
        })
    models = {}
    for model in sorted({d["model"] for d in details}):
        ds = [d for d in details if d["model"] == model]
        rows = [r for r in latest.values() if r["task"]["model_alias"] == model]
        attempts = [a for r in history if r["task"]["model_alias"] == model for a in r["attempts"]]
        known_costs = [((a.get("raw_response") or {}).get("usage") or {}).get("cost_rub") for a in attempts]
        models[model] = {
            "success": len(rows), "cases": len(ds),
            "majority_flips": sum(d["majority_changed"] for d in ds),
            "flip_rate": sum(d["majority_changed"] for d in ds) / len(ds),
            "repeatability_both": sum(d["equal_repeat_pairs"] for d in ds) / (6 * len(ds)),
            "repeatability_original": sum(x == y for d in ds for x, y in combinations(d["original"], 2)) / (3 * len(ds)),
            "repeatability_swapped": sum(x == y for d in ds for x, y in combinations(d["swapped_mapped"], 2)) / (3 * len(ds)),
            "cross_order_disagreement": sum(d["cross_disagree_pairs"] for d in ds) / (9 * len(ds)),
            "human_agreement_original": sum(d["human_agreement_original"] for d in ds) / len(ds),
            "human_agreement_count": sum(d["human_agreement_original"] for d in ds),
            "unanimous_original": sum(d["original_unanimous"] for d in ds),
            "unanimous_swapped": sum(d["swapped_unanimous"] for d in ds),
            "attempts": len(attempts), "failed_attempts": sum(a["status"] == "failed" for a in attempts),
            "known_cost_rub": sum(float(v) for v in known_costs if v is not None),
            "attempts_without_cost": sum(v is None for v in known_costs),
            "reasoning_tokens": 0 if model.startswith("qwen") else None,
            "by_collegium": {c: {
                "flips": sum(d["majority_changed"] for d in ds if d["collegium"] == c),
                "cases": sum(d["collegium"] == c for d in ds),
                "human_agreement": sum(d["human_agreement_original"] for d in ds if d["collegium"] == c),
            } for c in COLLEGIA},
            "flagged_bouts": {bid: [d["collegium"] for d in ds if d["bout_id"] == bid and d["majority_changed"]]
                              for bid in ("24", "27")},
        }
    with (OUT / "metrics/summary.csv").open(encoding="utf-8", newline="") as stream:
        inference = {r["test_id"]: r for r in csv.DictReader(stream) if r["test_id"]}
    flip_test = inference["H3:label_flip_rate_difference"]
    repeat_test = inference["H3:pairwise_repeatability_difference"]
    result = {
        "run_id": "015", "audited_utc": datetime.now(timezone.utc).isoformat(),
        "native_winners_preserved": True, "majority_flips_agree_with_standard_csv": True,
        "inputs_unchanged_since_pre_run_snapshot": True,
        "models": models, "cases": details,
        "inference_reference": {k: inference[k] for k in (
            "H2:accuracy_difference", "H3:label_flip_rate_difference", "H3:pairwise_repeatability_difference")},
    }
    (OUT / "independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    clef, qwen = models["clef"], models["qwen3-8-flash"]
    pct = lambda value: f"{value:.1%}".replace(".", ",")
    lines = [
        "# Основной эксперимент 015: Clef и Qwen 3.8 Flash", "",
        "Дата проведения: 25 октября 2026 года (Asia/Yekaterinburg).",
        "25 поединков, ID 6–30; три коллегии; original и swapped; три независимых голоса на каждое условие. 900/900 плановых голосов получены успешно.", "",
        "Обе модели оценивают полный поединок с общими содержательными промптами. Qwen: max_tokens=16, reasoning.effort=none; по журналу reasoning-токенов 0. Clef: штатный Decisions API. Все раунды передаются вместе. Канонизация и автоматическая коррекция победителя отсутствуют.", "",
        "Победитель коллегии — большинство трёх ответов. При сравнении ориентаций swapped переводится в исходных участников ровно один раз.", "",
        "| Показатель | Clef | Qwen 3.8 Flash |", "|---|---:|---:|",
        f"| Смена победителя после перестановки | {clef['majority_flips']}/75 ({pct(clef['flip_rate'])}) | {qwen['majority_flips']}/75 ({pct(qwen['flip_rate'])}) |",
        f"| Попарная повторяемость, original + swapped | {pct(clef['repeatability_both'])} | {pct(qwen['repeatability_both'])} |",
        f"| Попарная повторяемость, только original | {pct(clef['repeatability_original'])} | {pct(qwen['repeatability_original'])} |",
        f"| Расхождение всех пар между ориентациями | {pct(clef['cross_order_disagreement'])} | {pct(qwen['cross_order_disagreement'])} |",
        f"| Совпадение с большинством людей, original | {clef['human_agreement_count']}/75 ({pct(clef['human_agreement_original'])}) | {qwen['human_agreement_count']}/75 ({pct(qwen['human_agreement_original'])}) |",
        f"| Технические попытки, включая повторы ошибок | {clef['attempts']} | {qwen['attempts']} |",
        f"| Учтённая провайдером стоимость, ₽ | {clef['known_cost_rub']:.4f} | {qwen['known_cost_rub']:.4f} |", "",
        "## Смена победителя по коллегиям", "",
        "| Коллегия | Clef | Qwen 3.8 Flash |", "|---|---:|---:|",
    ]
    for c in COLLEGIA:
        lines.append(f"| {c} | {clef['by_collegium'][c]['flips']}/25 | {qwen['by_collegium'][c]['flips']}/25 |")
    lines += [
        "", "## Как читать результаты", "",
        "Устойчивость к переименованию и повторяемость одного входа — разные показатели. Высокая повторяемость не гарантирует устойчивость после перестановки или обоснованность решения конкретной коллегии. Совпадение с людьми приведено как отдельная характеристика, а не как доказанная правильность.", "",
        f"Наблюдаемая разница долей смены победителя — {float(flip_test['value']):.1f} п.п. (Clef − Qwen); статистически она не подтверждена после коррекции Holm: p={float(flip_test['adjusted_p_value']):.4f}, 95% описательный bootstrap CI {float(flip_test['ci_lower']):.2f}…{float(flip_test['ci_upper']):.2f} п.п. Значимое различие повторяемости относится к original: Clef 100%, Qwen {pct(qwen['repeatability_original'])}, Holm p={float(repeat_test['adjusted_p_value']):.4f}. Итоговое confirmed для H3 в штатном отчёте относится к совместной гипотезе «хотя бы один показатель устойчивости различается» и не означает значимость каждого её отдельного теста.", "",
        "75 случаев вложены в 25 поединков. Это наблюдаемые доли для данной выборки и конфигурации; их нельзя считать точной вероятностью смены победителя на любом новом поединке. Интервалы и статистические проверки находятся в штатном отчёте.", "",
        "В поединках 24 и 27 ранее обнаружены неоднозначные представления игроков в тексте первой реплики. Данные сохранены без правок, оба поединка включены в основной расчёт. Их влияние не установлено экспериментально. Основная выборка уже использовалась в прежнем запуске 011 и частично при диагностике, поэтому она не является полностью нетронутой тестовой выборкой. Запуск 011 с канонизацией методологически отличается от 015; прямое сравнение как эффекта одного изменения некорректно.", "",
        "## Проверка и файлы", "",
        "Родные ответы API независимо сверены с сохранёнными winner; смена большинства пересчитана из calls.jsonl и совпала со штатным CSV. Хеши кода, датасета, конфигурации и промптов совпали со снимком до первого API-вызова.", "",
        "- [Штатный отчёт с метриками](metrics/report.md)",
        "- [Все ответы и технические попытки](calls.jsonl)",
        "- [Результаты по каждой коллегии и поединку](metrics/per_bout_collegium.csv)",
        "- [Независимый пересчёт](independent_audit.json)",
        "- [Снимок до запуска](evaluation_snapshot.json)",
        "- [Проверка неизменности входов](snapshot_verification.json)",
        "- [Проверка полноты и перестановки в восстановленных запросах](request_payload_audit.json)",
    ]
    if clef["attempts_without_cost"] or qwen["attempts_without_cost"]:
        lines += ["", "Часть технических попыток не содержит cost_rub; указанная стоимость является суммой доступных значений."]
    (OUT / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    scripts = OUT / "diagnostic_scripts"
    scripts.mkdir(exist_ok=True)
    shutil.copy2(__file__, scripts / Path(__file__).name)
    driver = Path(__file__).with_name("run_main_current.py")
    shutil.copy2(driver, scripts / driver.name)
    print(json.dumps(models, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
