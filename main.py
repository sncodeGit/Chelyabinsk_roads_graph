# -*- coding: utf-8 -*-

# Необходимые модули
import pathlib

# Дополнительные скрипты
import config
import osm_parsing
import converting_to_csv

G, c_obj, c_nodes = osm_parsing.get_and_visual_graph()
graph_df = converting_to_csv.convert_to_csv(G, c_obj, c_nodes)