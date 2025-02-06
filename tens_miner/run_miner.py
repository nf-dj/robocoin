#!/usr/bin/env python3
"""Entry point script for the TensCoin miner."""

import os
import sys

# Add the parent directory to the Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from PyQt6.QtWidgets import QApplication
from tens_miner.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()