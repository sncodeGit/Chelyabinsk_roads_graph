# -*- coding: utf-8 -*-

# Необходимые модули
import pathlib

# Самописные пакеты
from roads_gigraph import graph
from roads_graph import functions as func

# Дополнительные скрипты
import config
import osm_parsing


# Читаем содержимое OSM-файла
with open(config.osm_file) as osm_file:
        osm = osm_file.read()