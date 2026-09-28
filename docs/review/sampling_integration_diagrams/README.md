# Sampling integration diagrams

Диаграммы для ревью последних изменений по интеграции Sampling Zoo в FEDOT.

Файлы:

- `sampling_components.puml` / `sampling_components.svg` - основные классы и связи.
- `fedot_fit_sampling_sequence.puml` / `fedot_fit_sampling_sequence.svg` - orchestration внутри `Fedot.fit`.
- `chunked_ensemble_composition_sequence.puml` / `chunked_ensemble_composition_sequence.svg` - обучение ансамбля по чанкам через общий validation holdout.
- `pipeline_ensemble_prediction_modes.puml` / `pipeline_ensemble_prediction_modes.svg` - режимы предсказания `PipelineEnsemble`.

Генерация через локальный PlantUML:

```bash
plantuml -tsvg docs/review/sampling_integration_diagrams/*.puml
```

Генерация через Python-клиент `plantuml` и PlantUML server:

```bash
.venv/bin/python -m plantuml -s http://www.plantuml.com/plantuml/svg/ docs/review/sampling_integration_diagrams/*.puml
```

Python-клиент может сохранить SVG с расширением `.png`; в таком случае файлы нужно переименовать в `.svg`.
