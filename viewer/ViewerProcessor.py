# -*- coding: utf-8 -*-
"""

@author: Steinn Ymir Agustsson

    Copyright (C) 2020 Steinn Ymir Agustsson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from processor import DldFlashDataframeCreator

class ViewerProcessor(DldFlashDataframeCreator.DldFlashProcessor):

    def __init__(self):
        super(ViewerProcessor, self).__init__()
        self.raw_data_path = self.DATA_RAW_DIR
        self.parquet_data_path = self.DATA_PARQUET_DIR
        self.binned_data_path = self.DATA_RESULTS_DIR
        self.runNumber = None

        self.pulseIdInterval = None
        self.path=None

    def is_valid_run_number(self):
        try:
            int(self.runNumber)
            return True
        except:
            return False

    def addBinning_test(self, name, start, end, steps):
        print(name, start, end, steps)

    def addFilter_test(self, colname, lb=None, ub=None):
            print(colname, lb, ub)

    def read_run(self):
        print(self.raw_data_path,
              self.parquet_data_path,
              self.binned_data_path,
              self.runNumber)
        # self.prc.readData(runNumber=self.runNumber,
        #                   pulseIdInterval=self.pulseIdInterval,
        #                   path=self.path)

def main():
    pass


if __name__ == '__main__':
    main()