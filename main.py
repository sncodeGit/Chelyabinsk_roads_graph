# -*- coding: utf-8 -*-

# Необходимые модули
import pathlib

# Самописные пакеты
import roads_graph.graph as graph

# Дополнительные скрипты
import config
import osm_parsing


# Читаем содержимое OSM-файла
with open(config.osm_file) as osm_file:
        osm = osm_file.read()

osm_parsing.get_roads(config.osm_file)