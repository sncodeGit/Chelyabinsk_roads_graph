# -*- coding: utf-8 -*-

# Необходимые модули
from geopy import distance

# Расчет кратчайшего географического расстояния между двумя точками по их широте и долготе
# Возвращает результат в км
def get_geo_distance(node1:object, node2:object):
    first = (node1.lat, node1.lon)
    second = (node2.lat, node2.lon)
    return distance.distance(first, second).km