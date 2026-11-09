# Памятка по сдаче

Срок сдачи понедельник 9 ноября, 16.59 по местному времени. Лучше сдать до 13.00.

## Что загрузить

| Поле формы | Откуда взять |
|---|---|
| PDF отчета | `report/Apart_report.pdf` |
| Аннотация | `report/abstract_for_form.txt`, текст совпадает с PDF слово в слово |
| Название | первая строка `report/content.md` после `% TITLE:` |
| Автор и место работы | строка `% AUTHOR:` в `report/content.md` |

После любой правки текста нужно пересобрать числа и PDF двумя командами.

```
uv run python experiments/build_report.py
uv run python experiments/render_official.py
```

Проверка правил спринта запускается так.

```
python3 ~/.claude/skills/sprint-win/scripts/check_rules.py report/Apart_report.pdf --form-abstract report/abstract_for_form.txt --sprint-start 2026-11-06
```

## Что нужно сделать тебе

1. Написать заявление об использовании LLM. Это раздел `## LLM Usage Statement` в конце `report/content.md`, сейчас там пустое место `{{llm_usage_statement}}`. Шаблон спринта требует коротко сказать, как использовались LLM.
2. Опубликовать репозиторий и вставить ссылку. Место `{{repo_url}}` в разделе Code and Data. Ссылку можно добавить в `report/numbers.json` ключом `repo_url` или прямо в текст.
3. Проверить строку автора и место работы.
4. По желанию запустить C5 на GPU. Это самый сильный способ поднять оценку, по симуляции судей около одного балла. Инструкция лежит в `docs/serve_organisms.md`, нужен под A100 80GB, около 1,19 $ в час на 4-6 часов. После подъема сервера все считается одной командой `caffeinate -i uv run python experiments/c5_run.py`. На том же сервере проверяются два организма на награде командой `caffeinate -i uv run python experiments/outcome_organism.py eval`.
5. По желанию активировать стек `swarm memory test` в приложении adaptionlabs.ai/app с включенной памятью. Тест общей памяти стоит центы и идет одной командой `uv run python experiments/adaption_memory_test.py --stack swarm-memory-test`.

## Что уже проверено

Основной текст занимает около 6,5 страницы из 8 разрешенных. Аннотация укладывается в 150-250 слов. Приложение «Limitations and Dual-Use / Ethical Considerations» на месте. Код до спринта раскрыт в `PRE_SPRINT_LOG.md`, README и отчете (раздел Code and Data, без путей к файлам). Все числа в тексте печатает `experiments/build_report.py`.

## Траты

Кредиты Adaption около 365 $ из одобренных 1 000 $. OpenRouter около 7 $.
