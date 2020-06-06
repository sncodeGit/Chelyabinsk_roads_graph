# -*- coding: utf-8 -*-

# Необходимые модули
import pathlib

# Дополнительные скрипты
import config
import osm_parsing
import converting_to_csv
import calculations
import time

G, c_obj, c_nodes = osm_parsing.get_and_visual_graph()
#graph_df = converting_to_csv.convert_to_csv(G, c_obj, c_nodes)
#graph_edges = converting_to_csv.df_to_edges(graph_df)
#n, a, b, c = calculations.get_path(G, graph_df, graph_edges, 1, c_obj, c_nodes)
#
#print('Номер узла(дома):')
#print(n)
#print('#'*64)
#print('Ближайший объект, расположенный по пути от узла(дома):')
#print(a)
#print('#'*64)
#print('Ближайший объект, расположенный по пути к узлу(дому):')
#print(b)
#print('#'*64)
#print('Ближайший объект, расположенный по пути "туда и обратно":')
#print(c)
#
#n, a, b, c = calculations.get_filtered_paths(G, graph_df, graph_edges, 3, c_obj, c_nodes, 3)
#
#print('Номер узла(дома):')
#print(n)
#print('#'*64)
#print('Объекты, расположенные менее, чем в Х км по пути от узла(дома):')
#print(a)
#print('#'*64)
#print('Объекты, расположенные менее, чем в Х км по пути к узлу(дому):')
#print(b)
#print('#'*64)
#print('Объекты, расположенные менее, чем в Х км по пути "туда и обратно":')
#print(c)
#
#start = time.monotonic()
#test = calculations.get_minmax_obj_path(G, graph_df, graph_edges, c_obj, c_nodes)
#print(time.monotonic() - start)
#print('Путь "туда": объект, самый дальний дом, расстояние')
#print(test['from'])
#print('#'*64)
#print('Путь "обратно": объект, самый дальний дом, расстояние')
#print(test['to'])
#print('#'*64)
#print('Путь "туда и обратно": объект, самый дальний дом, расстояние')
#print(test['sum'])
#
#
#calculations.get_shortest_paths_sum(G, graph_df, graph_edges, c_obj, c_nodes)
#
#calculations.get_lightest_tree(G, graph_df, graph_edges, c_obj, c_nodes)
