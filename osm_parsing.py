# -*- coding: utf-8 -*-

# Необходимые модули
import xml.etree.cElementTree as xml # Реализация на С для увеличения производительности
import pathlib

# Дополнительные скрипты
import config

# Создание отдельных XML-файлов с двусторонними дорогами, односторонними дорогами и зданиями
def make_part_files(osm_file, oneway_roads_file, twoway_roads_file, buildings_file):
    tree = xml.parse(osm_file)
    root = tree.getroot()

    # Корни деревьев, которые будут записаны в отдельные XML-файлы
    # Также добавляются символы переносы строк, для того, чтобы корневой тэг
    # не "слипся" с последующими тэгами, а также символ табуляции
    oneway_root = xml.Element("oneway")
    oneway_root.text = config.line_break + '\t'

    twoway_root = xml.Element("twoway")
    twoway_root.text = config.line_break + '\t'

    buildings_root = xml.Element("buildings")
    buildings_root.text = config.line_break + '\t'

    for way in root.findall('./way'):
        # highway - яаляется ли данный объект дорогой?
        # acceptable_road - допустим ли тип дороги?
        # oneway - односторонняя ли дорога?
        # building - является ли данный объект зданием?
        way_type = dict(highway = False, acceptable_road = False, oneway = False, building = False)

        for tag in way.findall('./tag'):
            # Дорога (любая)
            if tag.attrib['k'] == 'highway':
                way_type['highway'] = True
                # Определим тип дороги (будем ли добавлять её в файл)
                if tag.attrib['v'] in config.acceptable_road_types:
                    way_type['acceptable_road'] = True
            # Односторонняя дорога
            elif tag.attrib['k'] == 'oneway':
                way_type['oneway'] = True
            # Здание
            elif tag.attrib['k'] == 'building':
                way_type['building'] = True
        
        # В том случае, если данный объект будет записан в один из XML-файлов,
        # нужно заменить теги <nd> (ссылки) на объекты <node>
        counter = 0
        if True in way_type.values():
            for element in way:
                if element.tag == 'nd':
                    ref_id = element.attrib['ref']
                    node = root.find("./node[@id='%s']" % ref_id)
                    way[counter] = node
                counter += 1
        
        if way_type['building']:
            buildings_root.append(way)
        
        if way_type['acceptable_road']:
            if way_type['oneway']:
                oneway_root.append(way)
            else:
                twoway_root.append(way) 
        
    # Создадим деревья из сохраненных элементов и запишем эти деревья в файл
    tree = xml.ElementTree(buildings_root)
    tree.write(buildings_file)

    tree = xml.ElementTree(oneway_root)
    tree.write(oneway_roads_file)

    tree = xml.ElementTree(twoway_root)
    tree.write(twoway_roads_file)
        
# Запуск для парсинга основного OSM-файла с выделелением дорог и зданий в отедльные файлы
if __name__ == '__main__':
    make_part_files(config.osm_file, config.oneway_roads_file, config.twoway_roads_file, config.buildings_file)
    print('Files were created')