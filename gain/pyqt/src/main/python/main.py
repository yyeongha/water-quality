import sys
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class Interface:
    def main(self):
        self.window = QWidget()
        self.window.setGeometry(550, 300, 850, 550)

        debug_label_1 = QLabel('debug label 1')
        debug_label_1.setStyleSheet("border: 1px solid red;")

        debug_label_2 = QLabel('debug label 2')
        debug_label_2.setStyleSheet("border: 1px solid red;")

        debug_label_3 = QLabel('debug label 3')
        debug_label_3.setStyleSheet("border: 1px solid red;")

        debug_label_4 = QLabel('debug label 4')
        debug_label_4.setStyleSheet("border: 1px solid red;")

        debug_label_5 = QLabel('debug label 5')
        debug_label_5.setStyleSheet("border: 1px solid red;")

        debug_label_6 = QLabel('debug label 6')
        debug_label_6.setStyleSheet("border: 1px solid red;")

        debug_label_7 = QLabel('debug label 7')
        debug_label_7.setStyleSheet("border: 1px solid red;")

        debug_label_8 = QLabel('debug label 8')
        debug_label_8.setStyleSheet("border: 1px solid red;")

        grid_layout = QGridLayout() 
        
        # row, col, row_span, col_span
        
        # top
        grid_layout.addWidget(debug_label_1, 0, 0, 1, 2)

        # left
        grid_layout.addWidget(debug_label_2, 1, 0, 4, 1)
        grid_layout.addWidget(debug_label_3, 5, 0, 3, 1)

        # right
        grid_layout.addWidget(debug_label_4, 1, 1, 1, 1)
        grid_layout.addWidget(debug_label_5, 2, 1, 1, 1)
        grid_layout.addWidget(debug_label_6, 3, 1, 2, 1)
        grid_layout.addWidget(debug_label_7, 5, 1, 1, 1)
        grid_layout.addWidget(debug_label_8, 6, 1, 2, 1)
       
        self.window.setLayout(grid_layout)
        self.window.show()


if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    interface = Interface()
    interface.main()
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)