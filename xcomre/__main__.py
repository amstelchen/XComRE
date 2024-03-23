#!/usr/bin/env python

# XComRE
# # GPL License
# Copyright (c) 2024 Michael John <michael.john@gmx.at>

import sys
import os
import warnings
import urllib3
import shutil
import argparse
import logging
import configparser
import struct
from collections import namedtuple
from tabulate import tabulate
import pandas as pd
import numpy as np

from xcomre import PROGNAME, VERSION, AUTHOR, DESC


class CapitalisedHelpFormatter(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = 'Usage: '
        return super(CapitalisedHelpFormatter, self).add_usage(
            usage, actions, groups, prefix)

config = configparser.ConfigParser()

def main():
    parser = argparse.ArgumentParser(prog=PROGNAME, description=DESC, add_help=True, formatter_class=CapitalisedHelpFormatter)
    parser.add_argument('-v', '--version', action='version', 
        version='%(prog)s ' + VERSION + ' ' + AUTHOR)
    parser.add_argument('-l', '--list', action='store_true', 
        help='list game folders in use', required=False)
    parser.add_argument('--game', dest='game', choices=['XCOM', 'TFTD'],
        help='choose a game [XCOM (default)|TFTD]', type=str, required=False)
    parser.add_argument('--savegame', dest='savegame', 
        help='list a game folder\'s contents, i.e. `1` or `6`', type=str, required=False)
    parser.add_argument('--conffile', dest='conffile', 
        help='use custom configuration file', type=str, required=False)
    parser.add_argument('--format', dest='format',
        help='output format [tab (default)|pandas|json|csv]', type=str)

    args = parser.parse_args()
    game = vars(args)["game"] or "XCOM"
    savegame = vars(args)["savegame"]
    format = vars(args)["format"] or "tab"
    print(str(args).replace("Namespace", "Options"))

    global config    
    if vars(args)["conffile"]:
        config.read(os.path.abspath(vars(args)["conffile"]))
    else:
        config.read(os.path.join(os.path.dirname(__file__), "xcomre.conf"))
    # print(config.get("General", "game_path"))
    print(config[game]['game_path'])

    if vars(args)["list"]:
        list_files(format)

    if vars(args)["savegame"]:
        decode_files(game, savegame, format)


def list_files(format: str):
    url = r'https://www.ufopaedia.org/index.php/Game_Files'
    filename = 'Game_Files'

    if not os.path.exists(filename):
        # Creating a PoolManager instance for sending requests.
        http = urllib3.PoolManager()
        # Sending a GET request and getting back response as HTTPResponse object.
        with open(filename, 'wb') as out:
            r = http.request('GET', url, preload_content=False)
            shutil.copyfileobj(r, out)

    warnings.filterwarnings("ignore",category=FutureWarning)
    pd.set_option('display.max_colwidth', None)
    #pd.set_option('max_colwidth', 800)
    pd.set_option("display.precision", 0)

    tables = pd.read_html(filename) # url) #, index_col=0) # Returns list of all tables on page
    ProgramFilesTable = tables[0]   # noqa: F841
    GeoscapeFilesTable = tables[1]
    BattlescapeFilesTable = tables[2]  # noqa: F841
    #tables[1].style.set_properties(**{'text-align': 'left'})
    #print(tables[1].to_string(justify='left'))

    #GeoscapeFilesTable.loc[20, "Record Format"] = "H26sHHHHHH"

    #GeoscapeFilesTable.iloc["Record Format"].dtype = np.string_
    #GeoscapeFilesTable['File'] = GeoscapeFilesTable['File'].str.replace('(TFTD)', '')

    #GeoscapeFilesTable.loc[GeoscapeFilesTable['File'] == 'SAVEINFO.DAT', "Record Format"] = "H26sHHHHHH"
    #GeoscapeFilesTable.loc[GeoscapeFilesTable['File'] == 'LIGLOB.DAT', "Record Format"] = 'i12i12i12i'
    #GeoscapeFilesTable.loc[GeoscapeFilesTable['File'] == 'LOC.DAT', "Record Format"] = 'bbHHHHHHbbI' 
    #GeoscapeFilesTable.loc[GeoscapeFilesTable['File'] == 'TRANSFER.DAT', "Record Format"] = 'BBBBcB' 

    Mapping = pd.DataFrame({
        'File': ['ACTS.DAT', 'ASTORE.DAT', 'SAVEINFO.DAT', 'LIGLOB.DAT', 'LOC.DAT', 'TRANSFER.DAT'], 
        'Record Format': ["7c", "bbb9B", "H26sHHHHHH", 'i12i12i12i', 'bbHHHHHHbbI', 'BBBBcB'],
        #'Description': [],
        'Record Length': [7, 12, 40, 144, 20, 8],
        'Total Records': [12, 50, 1, 1, 50, 100]
    })
    Mapping['Record Length'] = Mapping['Record Length'].astype('int8')
    Mapping['Total Records'] = Mapping['Total Records'].astype('int')

    print(Mapping.dtypes)

    #GeoscapeFilesTable.concat(Mapping)
    #GeoscapeFilesTable.join(Mapping)
    GeoscapeFilesTable = pd.merge(GeoscapeFilesTable, Mapping, how='left')

    #GeoscapeFilesTable.insert("SAVEINFO.DAT", "D", 5)
    #GeoscapeFilesTable.insert(20, "D", 1)

    GeoscapeFilesTable = GeoscapeFilesTable.replace([np.nan, -np.inf], '')
    #GeoscapeFilesTable['Record Length'] = GeoscapeFilesTable['Record Length'].astype('int')

    #GeoscapeFilesTable['Total Size']
    #GeoscapeFilesTable = GeoscapeFilesTable.assign(F = GeoscapeFilesTable['Record Length'] * 10)
    #GeoscapeFilesTable.loc[:, "Total Size"] = GeoscapeFilesTable['Record Length'] * GeoscapeFilesTable['Record Length']
    #GeoscapeFilesTable.loc[:, "Total Size"] = GeoscapeFilesTable['Record Length'] * GeoscapeFilesTable['Total Records']
    #GeoscapeFilesTable['Description'] = GeoscapeFilesTable['Description'].str.split('.')[0]

    #print(GeoscapeFilesTable.dtypes)
    #print(GeoscapeFilesTable)

    GeoscapeFilesTable = GeoscapeFilesTable.loc[:, GeoscapeFilesTable.columns != 'Description']

    if format == "tab":
        print(tabulate(GeoscapeFilesTable, maxcolwidths=140, showindex=False, headers=GeoscapeFilesTable.columns, tablefmt='presto'))
    if format == "json":
        print(GeoscapeFilesTable.to_json(orient='values', index=None))
    if format == "csv":
        print(GeoscapeFilesTable.loc[:, GeoscapeFilesTable.columns != 'Description'].to_csv(index=None))

facilities = {
    0x00: "Access Lift",
    0x01: "Living Quarters",
    0x02: "Laboratory",
    0x03: "Workshop",
    0x04: "Small Radar System",
    0x05: "Large Radar System",
    0x06: "Missile Defense",
    0x07: "General Stores",
    0x08: "Alien Containment",
    0x09: "Laser Defense",
    0x0A: "Plasma Defense",
    0x0B: "Fusion Ball Defense",
    0x0C: "Grav Shield",
    0x0D: "Mind Shield",
    0x0E: "Psionic Laboratory",
    0x0F: "Hyper-wave Decoder",
    0x10: "Hangar (Top Left)",
    0x11: "Hangar (Top Right)",
    0x12: "Hangar (Bottom Left)",
    0x13: "Hangar (Bottom Right)",
    0xFF: "Empty"
}

abbreviations = []

# Iterate over the facility names
for name in facilities.values():
    #words = name.split()  # Split the facility name into words
    #for word in words:
    abbreviation = ''.join(filter(str.isupper, name))  # Extract uppercase letters
    if abbreviation:  # Check if abbreviation is not empty
        abbreviations.append(abbreviation)

abbreviations = ['AL', 'LQ', 'LA', 'WS', 'SR', 'LR', 'MD', 'GS', 'AC', 'LD', 'PD', 'FB', 'GR', 'MS', 'PL', 'HD', 'HTL', 'HTR', 'HBL', 'HBR', '  ']
abbreviations = ['AL', 'LQ', 'LA', 'WS', 'SR', 'LR', 'MD', 'GS', 'AC', 'LD', 'PD', 'FB', 'GR', 'MS', 'PL', 'HD', 'HA', 'HA', 'HA', 'HA', '  ']
# print(abbreviations)
# print(dict(map(lambda x: x.split(': '), facilities.split('\n'))))

def read_records_from_binary_file(file_path, record_format, record_name, field_names):
    Record = namedtuple(record_name.replace('.', '_') or 'Record', field_names)  # Adjust field names as per your format
    records = []
    record_size = struct.calcsize(record_format)

    with open(file_path, 'rb') as file:
        while True:
            record_data = file.read(record_size)
            if len(record_data) < record_size:
                break  # Reached end of file
            record = Record._make(struct.unpack(record_format, record_data))
            records.append(record)
    return records

def decode_files(game, game_number, format):
    global config

    # Example usage:
    file_path = 'binary_data.dat'
    file_path = config[game]['game_path'] + 'GAME_' + game_number + '/'

    if not os.path.exists(file_path + 'SAVEINFO.DAT'):
        print(f"{file_path} does not exist!")
        return

    file_name = 'ACTS.DAT'
    record_format = '7c' 
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, 
        field_names=['Research', 'Harvest', 'Abduction', 'Infiltration', 'Base', 'Terror', 'Retaliation'])
    print(records)

    file_name = 'ASTORE.DAT'
    record_format = 'bbb9s' 
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, 
        field_names=['Alien_race', 'Rank', 'Base', 'unclear'])
    #for record in records:
    #    print(record.Alien_race, record.Rank, record.Base)
    print(records)

    file_name =  'BASE.DAT' # 'SOLDIER.DAT'
    record_format = '<16sHHH270s' # 292 Bytes gesamt
    record_format = '<16sHHH36s234s' 
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, field_names=[
        'BaseName', 
        'short_range_detection_capability', 
        'long_range_detection_capability', 
        'hyperwave_detection_capability',
        'facilities',
        'unused'])
    for record in records:
        print(record.BaseName.decode("utf-8"))
        #print(record.BaseName.decode("utf-8"),
        #    record.short_range_detection_capability, 
        #    record.long_range_detection_capability,
        #    record.hyperwave_detection_capability,
        #    record.facilities)

        for i, facility in enumerate(record.facilities, 1):
            if facility == 255:
                print("  ", end=" ")
            else:
                print(abbreviations[facility], end=" ")
            if i % 6 == 0:
                print("")
            else:
                print("|", end=" ")


    file_name = 'SAVEINFO.DAT'
    record_format = 'H26sHHHHHH' 
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, 
        field_names=['direct', 'save_name', 'year', 'month', 'day', 'hour', 'minute', 'type'])
    print(records)

    file_name = 'LIGLOB.DAT'
    record_format = 'i12i12i12i' 
    fields = ['money']
    fields += ['Expenditure_' + str(x) for x in range(1, 12 + 1)]
    fields += ['Maintenance_' + str(x) for x in range(1, 12 + 1)]
    fields += ['Balance_' + str(x) for x in range(1, 12 + 1)]
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, 
        field_names=fields)
    print(records)

    file_name = 'LOC.DAT'
    record_format = 'bbHHHHHHbbI' 
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, 
        field_names=['type', 'reference', 'longitude', 'latitude', 'countdown', 'fractional', 'count_suffix', 'UNUSED1', 'craft_transfer_mode', 'UNUSED2', 'visiblity_mobility'])
    # print(records[0])

    file_name = 'TRANSFER.DAT'
    record_format = 'BBBBcB' 
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, 
        field_names=['from_base', 'to_base', 'hours', 'type', 'reference', 'quantity'])
    # print(records)

    """file_name = 'FACIL.DAT'
    record_format = '16B' 
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, 
        field_names=[''])
    # print(records)

    file_name = 'INTER.DAT'
    record_format = '30B6I' 
    records = read_records_from_binary_file(file_path + file_name, record_format, file_name, 
        field_names=[''])
    # print(records)"""

def read_numpy_data(game, game_number, format):
    global config

    import numpy as np
    import fnmatch
    import os

    file_path = config[game]['game_path'] + 'GAME_' + game_number + '/'
    # file_path = '/nvme/SteamLibrary/steamapps/common/XCom UFO Defense/XCOM/GAME_1/'

    """for file in os.listdir(file_path):
        if fnmatch.fnmatch(file, 'GAME_', ):
            print(file)

    import glob
    file_names = file_path + 'GAME_*/' + 'IGLOB.DAT'
    files = sorted(glob.glob(file_path + 'GAME_*'))
    #print(sorted(glob.glob(file_names)))
    for file in files:
        dt = np.dtype([('time', [('min', np.int32), ('sec', np.int32)]),
            ('temp', float)])

        records = np.fromfile(file_path + file_name, dtype=dt)
        """
    file_name = 'IGLOB.DAT'
    dt = np.dtype([('time', [('min', np.int32), ('sec', np.int32)]), ('temp', float)])
    file_name = 'SAVEINFO.DAT'
    dt = np.dtype([('time', [('min', np.int32), ('sec', np.int32)]), ('temp', float)])
    records = np.fromfile(file_path + file_name, dtype=dt)
    print(records)


if __name__ == '__main__':
    # read_numpy_data('XCOM', 1, '')
    main()
