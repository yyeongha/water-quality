import sys
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QTextEdit, QDialog)


class Interface:
    def main(self):
        self.window = QWidget()
        # self.window.setGeometry(550, 300, 850, 550)

        myDialog = QDialog()
        myDialog.setFixedSize(500, 500)
 
        # below write code
        grid = QGridLayout()
        self.window.setLayout(grid)

        layout1 = QLabel('debug layout')
        grid.addWidget(layout1, 0, 0, 1, 2)
        layout1.setStyleSheet("border: 1px solid red;") 

        layout1 = QLabel('debug layout')
        grid.addWidget(layout1, 1, 0)
        layout1.setStyleSheet("border: 1px solid red;") 

        layout1 = QLabel('debug layout')
        grid.addWidget(layout1, 1, 1)
        layout1.setStyleSheet("border: 1px solid red;") 
        
        self.window.show()


if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    interface = Interface()
    interface.main()
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)