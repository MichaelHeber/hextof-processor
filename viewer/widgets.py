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

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import QMainWindow, QDoubleSpinBox, \
    QRadioButton, QLineEdit, QComboBox, QSizePolicy, \
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QPushButton, QGridLayout, QSpinBox, QLabel, QFrame
from pyqtgraph.Qt import QtCore as pQtCore, QtGui as pQtGui


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


    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QHBoxLayout()
        central_widget.setLayout(central_layout)

        self.input_box = QGroupBox('Select Run')



def main():
    pass


if __name__ == '__main__':
    main()