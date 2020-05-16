# -*- coding: utf-8 -*-

# Базовый дескриптор типа узла
# Значение - тип точки неизвестен
class NodeType(object):
	def __init__(self):
		self.type = None

# Дескриптор типа узла
# Значение - точка является составляющей дороги
class Road(NodeType):
	def __init__(self):
		self.num = 1

# Дескриптор типа узла
# Значение - точка является зданием
class Building(NodeType):
	def __init__(self):
		self.num = 2

# Основной объект класса
# Содержит георгафические координаты, соседей и дескриптор типа (NodeType)
class Node(object):
	def __init__(self, lat:float, lon:float, node_type:NodeType, out_nodes:list = None, in_nodes:list = None):
		self.lat = lat
		self.lon = lon
		self.out_nodes = out_nodes
		self.in_nodes = in_nodes
		self.type = node_type