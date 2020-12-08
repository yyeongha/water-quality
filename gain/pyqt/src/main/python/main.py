import sys
import matplotlib.pyplot as plt
import random 
import pandas as pd

from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 


df = pd.DataFrame({'a': ['Mary', 'Jim', 'John'],
                   'b': [100, 200, 300],
                   'c': ['a', 'b', 'c']})

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
        
class Interface:
    def main(self):
        self.window = QWidget()
        self.window.setGeometry(550, 300, 850, 550)


        ''' widget '''
        # layout 1
        label_title = QLabel('TOC 예측 시스템')
        label_title.setStyleSheet("border: 1px solid red;")

        # layout 2
        # debug_label_2 = QLabel('debug label 2')
        # debug_label_2.setStyleSheet("border: 1px solid red;")
        debug_label_2 = QCalendarWidget()

        # layout 3
        # debug_label_3 = QLabel('debug label 3')
        # debug_label_3.setStyleSheet("border: 1px solid red;")
        model = pandasModel(df)
        view = QTableView()
        view.setModel(model)
        debug_label_3 = view

        # layout 4
        debug_label_4_1 = QLabel('학습기간')
        debug_label_4_1.setStyleSheet("border: 1px solid red;")

        debug_label_4_2 = QLineEdit()

        debug_label_4_3 = QLabel('~')
        debug_label_4_3.setStyleSheet("border: 1px solid red;")

        debug_label_4_4 = QLineEdit()

        debug_label_4_5 = QLabel('(2주간)')
        debug_label_4_5.setStyleSheet("border: 1px solid red;")
        # debug_label_4 = QLineEdit()

        # layout 5
        debug_label_5_1 = QLabel('예측기간')
        debug_label_5_1.setStyleSheet("border: 1px solid red;")

        debug_label_5_2 = QLineEdit()

        debug_label_5_3 = QLabel('~')
        debug_label_5_3.setStyleSheet("border: 1px solid red;")

        debug_label_5_4 = QLineEdit()

        debug_label_5_5 = QLabel('(3일간)')
        debug_label_5_5.setStyleSheet("border: 1px solid red;")

        # layout 6
        # debug_label_6 = QLabel('debug label 6')
        # debug_label_6.setStyleSheet("border: 1px solid red;")
        debug_label_6_1 = QLabel('예측 그래프')
        debug_label_6_1.setStyleSheet("border: 1px solid red;")

        data = [random.random() for i in range(10)] 
        figure = plt.figure() 
        ax = figure.add_subplot(111)
        ax.plot(data, '*-') 
        debug_label_6_2 = FigureCanvas(figure)

        # layout 7
        debug_label_7_1 = QLabel('예측 수질 (3일간)')
        debug_label_7_1.setStyleSheet("border: 1px solid red;")

        debug_label_7_2_1 = QLabel('debug label 7 2 1')
        debug_label_7_2_1.setStyleSheet("border: 1px solid red;")

        debug_label_7_2_2 = QLabel('debug label 7 2 2')
        debug_label_7_2_2.setStyleSheet("border: 1px solid red;")

        debug_label_7_2_3 = QLabel('debug label 7 2 3')
        debug_label_7_2_3.setStyleSheet("border: 1px solid red;")

        debug_label_7_3_1 = QLabel('debug label 7 3 1')
        debug_label_7_3_1.setStyleSheet("border: 1px solid red;")

        debug_label_7_3_2 = QLabel('debug label 7 3 2')
        debug_label_7_3_2.setStyleSheet("border: 1px solid red;")

        debug_label_7_3_3 = QLabel('debug label 7 3 3')
        debug_label_7_3_3.setStyleSheet("border: 1px solid red;")

        # layout 8
        # debug_label_8 = QLabel('debug label 8')
        # debug_label_8.setStyleSheet("border: 1px solid red;")
        debug_label_8_1 = QLabel('예측 강우량 및 기온')
        debug_label_8_1.setStyleSheet("border: 1px solid red;")

        data = [random.random() for i in range(10)] 
        figure = plt.figure() 
        ax = figure.add_subplot(111)
        ax.plot(data, '*-') 
        debug_label_8_2 = FigureCanvas(figure)


        ''' layout '''
        grid_layout = QGridLayout() 
        
        # row, col, row_span, col_span
        
        # top
        grid_layout.addWidget(label_title, 0, 0, 1, 7)

        # left
        gab = 3
        grid_layout.addWidget(debug_label_2, 1, 0, 4, 1)
        grid_layout.addWidget(debug_label_3, 5, 0, 3, 1)

        # right
        grid_layout.addWidget(debug_label_4_1, 1, 1, 1, 2)
        grid_layout.addWidget(debug_label_4_2, 1, 3, 1, 1)
        grid_layout.addWidget(debug_label_4_3, 1, 4, 1, 1)
        grid_layout.addWidget(debug_label_4_4, 1, 5, 1, 1)
        grid_layout.addWidget(debug_label_4_5, 1, 6, 1, 1)

        grid_layout.addWidget(debug_label_5_1, 2, 1, 1, 2)
        grid_layout.addWidget(debug_label_5_2, 2, 3, 1, 1)
        grid_layout.addWidget(debug_label_5_3, 2, 4, 1, 1)
        grid_layout.addWidget(debug_label_5_4, 2, 5, 1, 1)
        grid_layout.addWidget(debug_label_5_5, 2, 6, 1, 1)

        grid_layout.addWidget(debug_label_6_1, 3, 1, 1, 6)
        grid_layout.addWidget(debug_label_6_2, 4, 1, 1, 6)

        grid_layout.addWidget(debug_label_7_1, 5, 1, 1, 6)

        grid_layout.addWidget(debug_label_7_2_1, 6, 1, 1, 2)
        grid_layout.addWidget(debug_label_7_2_2, 6, 3, 1, 2)
        grid_layout.addWidget(debug_label_7_2_3, 6, 5, 1, 2)
        
        grid_layout.addWidget(debug_label_7_3_1, 7, 1, 1, 2)
        grid_layout.addWidget(debug_label_7_3_2, 7, 3, 1, 2)
        grid_layout.addWidget(debug_label_7_3_3, 7, 5, 1, 2)
        
        grid_layout.addWidget(debug_label_8_1, 8, 1, 1, 6)
        grid_layout.addWidget(debug_label_8_2, 9, 1, 1, 6)
       

        self.window.setLayout(grid_layout)
        self.window.show()


if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    interface = Interface()
    interface.main()
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)