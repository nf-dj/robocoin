"""Mining control panel with input fields and buttons."""

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout,
                           QLabel, QLineEdit, QPushButton, QComboBox)
import torch
from ..mining import BATCH_SIZE

class MiningControls(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Node connection
        node_layout = QHBoxLayout()
        node_label = QLabel("Node:")
        self.node_input = QLineEdit()
        self.node_input.setText("65.109.105.241:8332")
        node_layout.addWidget(node_label)
        node_layout.addWidget(self.node_input)
        layout.addLayout(node_layout)

        # Mining address
        address_layout = QHBoxLayout()
        address_label = QLabel("Mining Address:")
        self.address_input = QLineEdit()
        self.generate_address_button = QPushButton("Generate New")
        address_layout.addWidget(address_label)
        address_layout.addWidget(self.address_input)
        address_layout.addWidget(self.generate_address_button)
        layout.addLayout(address_layout)

        # Batch size input
        batch_layout = QHBoxLayout()
        batch_label = QLabel("Batch Size:")
        self.batch_input = QLineEdit()
        self.batch_input.setText(str(BATCH_SIZE))
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_input)
        layout.addLayout(batch_layout)

        # Device selection
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "mps", "cpu"])
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

        # Auto-detect best device
        if torch.cuda.is_available():
            self.device_combo.setCurrentText("cuda")
        elif torch.backends.mps.is_available():
            self.device_combo.setCurrentText("mps")
        else:
            self.device_combo.setCurrentText("cpu")

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Mining")
        self.stop_button = QPushButton("Stop")
        self.test_button = QPushButton("Run Test")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.test_button)
        layout.addLayout(button_layout)

    def get_node_address(self):
        """Get node address and port."""
        addr = self.node_input.text().strip()
        try:
            host, port = addr.split(":")
            return host, int(port)
        except ValueError:
            return None, None

    def get_batch_size(self):
        """Get batch size as integer."""
        try:
            size = int(self.batch_input.text())
            return size if size > 0 else None
        except ValueError:
            return None

    def set_mining_state(self, is_mining):
        """Update controls based on mining state."""
        self.start_button.setEnabled(not is_mining)
        self.test_button.setEnabled(not is_mining)
        self.stop_button.setEnabled(is_mining)
        self.node_input.setEnabled(not is_mining)
        self.batch_input.setEnabled(not is_mining)
        self.device_combo.setEnabled(not is_mining)
        self.address_input.setEnabled(not is_mining)
        self.generate_address_button.setEnabled(not is_mining)