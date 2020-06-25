# -*- coding: utf-8 -*-
"""

@author: Steinn Ymir Agustsson

    Copyright (C) 2018 Steinn Ymir Agustsson

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
import sys, os
import logging

import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import QMainWindow, QDoubleSpinBox, \
    QRadioButton, QLineEdit, QComboBox, QSizePolicy, \
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QPushButton, QGridLayout, QSpinBox, QLabel, QFrame
from pyqtgraph.Qt import QtCore as pQtCore, QtGui as pQtGui

from viewer.ViewerProcessor import ViewerProcessor



class ViewerMainWindow(QMainWindow):

    def __init__(self):
        super(ViewerMainWindow, self).__init__()
        self.logger = logging.getLogger('{}.FastScanMainWindow'.format(__name__))
        self.logger.info('Created MainWindow')

        self.setWindowTitle('Hextof Viewer')
        self.setGeometry(100, 50, 1152, 768)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('ready')

        # set the cool dark theme and other plotting settings
        try:
            import qdarkstyle
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            pg.setConfigOption('background', (25, 35, 45))
            pg.setConfigOption('foreground', 'w')
            pg.setConfigOptions(antialias=True)
        except:
            pass

        self.processor = ViewerProcessor()


        self.setup_ui()



    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)

        control_widget = self.make_control_widget()
        control_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        visual_widget = self.make_visual_widget()
        visual_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        main_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(control_widget)
        main_splitter.addWidget(visual_widget)
        # main_splitter.setStretchFactor(0, 5)

        central_layout.addWidget(main_splitter)

    def make_control_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(self.make_data_select_box())
        layout.addStretch()
        return widget

    def make_data_select_box(self):
        widget = QGroupBox('Select Data')
        layout = QGridLayout()
        widget.setLayout(layout)

        self.raw_data_path_ledit = QLineEdit(self.processor.DATA_RAW_DIR)
        layout.addWidget(QLabel('Raw data:'), 0, 0)
        layout.addWidget(self.raw_data_path_ledit, 0, 1)
        self.raw_data_path_ledit.editingFinished.connect(self.set_raw_data_path)

        self.parquet_data_path_ledit = QLineEdit(self.processor.DATA_PARQUET_DIR)
        layout.addWidget(QLabel('Parquet data:'), 1, 0)
        layout.addWidget(self.parquet_data_path_ledit, 1, 1)
        self.parquet_data_path_ledit.editingFinished.connect(self.set_parquet_data_path)


        self.binned_data_path_ledit = QLineEdit(self.processor.DATA_RESULTS_DIR)
        layout.addWidget(QLabel('Binned data:'), 2, 0)
        layout.addWidget(self.binned_data_path_ledit, 2, 1)
        self.binned_data_path_ledit.editingFinished.connect(self.set_binned_data_path)

        self.run_number_ledit = QLineEdit(None)
        layout.addWidget(QLabel('Run Number:'), 3, 0)
        layout.addWidget(self.run_number_ledit, 3, 1)
        self.run_number_ledit.editingFinished.connect(self.set_run_number)


        self.read_run_button = QPushButton('Read Raw Data')
        layout.addWidget(self.read_run_button, 4, 0, 2, 1)
        self.read_run_button.clicked.connect(self.processor.read_run)
        self.read_run_button.setEnabled(self.processor.is_valid_run_number())


        return widget

    def make_visual_widget(self):
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        layout.addStretch()
        return widget

    @QtCore.pyqtSlot()
    def set_raw_data_path(self):
        self.processor.DATA_RAW_DIR = self.raw_data_path_ledit.text()

    @QtCore.pyqtSlot()
    def set_parquet_data_path(self):
        self.processor.DATA_PARQUET_DIR = self.parquet_data_path_ledit.text()

    @QtCore.pyqtSlot()
    def set_binned_data_path(self):
        self.processor.DATA_RESULTS_DIR = self.binned_data_path_ledit.text()

    @QtCore.pyqtSlot()
    def set_run_number(self):
        self.processor.runNumber = self.run_number_ledit.text()
        self.read_run_button.setEnabled(self.processor.is_valid_run_number())

    # @QtCore.pyqtSlot
    # def set_raw_data_path(self, s):
    #     self.processor.raw_data_path = s


def main():
    pass


if __name__ == '__main__':
    main()