# Chelyabinsk_roads_graph
Проект по курсу "Теория конечных графов и её приложения"

## OSM-файл, который используется в проекте: 
https://drive.google.com/file/d/1G6H4iq8ShN9-w8atgrdN6-1Vje6suVce/view?usp=sharing

Необходимо расположить в директории `OSM/`

## Этап предобработки OSM-файла
Для того, чтобы избежать необходимости каждый раз парсить OSM-файл, было принято решение 
реализовать этап предобработки данных. Основная проблема при отсутствии предобработки заключается в том, что в тегах \<way\>
точки (\<node\>) представляются как ссылки на соответствующие теги \<node\> (\<nd\>). Сначала выделяются объекты \<way\>, которые представляют интерес в рамках проекта, и записываются в соответствующие файлы в директорию `OSM/`. 
Подробное описание файлов можно найти в разделе **Файловая структура проекта**

Для запуска предобработки файла необходимо выполнить `python3 osm_parsing.py`

## Файловая структура проекта

### `OSM/`
Содержит исходный OSM-файл, а также созданные при предобработке XML-файлы:

1) `building.xml` - файл с выделенными зданиями

  https://drive.google.com/open?id=1YaofO0lyec7W-wol_whFBxxi2icR4JSn

2) `oneway_roads.xml` - файл с выделенными односторонними дорогами

  https://drive.google.com/open?id=1OAUjBbPIEOnDBb-xvgrp8rs-iMqILMuW

3) `twoway_roads.xml` - файл с выделенными двусторонними дорогами

  https://drive.google.com/open?id=1ByOh-a-CbPtHSM379GJu-4aTun8gKZCX

**Важно:** *Структура данных в этих данных сохранена относительно формата OSM, но узлы \<nd\> (ссылки на \<node\>) заменены на \<node\>*

### `roads_graph/`
Пакет для работы с графом (класс + методы)

1) `graph.py` - описание класса графа
2) `graph_functions.py` - базовые функции работы с объектами класса графа

### `Docs/`
1) `Docs/requirements.txt` - необходимые для запуска проекта модули Python
2) `Docs/img/*` - изображения, сохраненные для дальнейшего использования при презентации проекта

### `osm_parsing.py`
Парсинг OSM-файла

### `config.py`
Файл с переменными непакетной части проекта
