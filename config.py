# -*- coding: utf-8 -*-

import pathlib

# Путь до OSM-файла
osm_file = pathlib.Path.cwd() / 'OSM' / 'Chelyabinsk.osm'

"""
Переменные предобработки данных (создание специальных файлов)
"""

### Пути до выделенных при предобработке частей OSM-файла
# Двусторонние дороги
twoway_roads_file = pathlib.Path.cwd() / 'OSM' / 'twoway_roads.xml'
# Односторонние дороги
oneway_roads_file = pathlib.Path.cwd() / 'OSM' / 'oneway_roads.xml'
# Здания
buildings_file = pathlib.Path.cwd() / 'OSM' / 'buildings.xml'

# Список типов дорог, которые являются "допустимыми"
acceptable_road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential']
acceptable_road_types += ['motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']
acceptable_road_types += ['living_street', 'service']

# Перенос строки, используемый при создании дополнительных XML-файлов
line_break = '\r\n'