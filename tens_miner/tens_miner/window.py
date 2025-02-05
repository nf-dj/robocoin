"""Main window implementation for the TensCoin miner GUI."""

import json
import math
import torch
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QLineEdit, QPushButton, QComboBox,
                           QTextEdit, QFrame, QGridLayout, QMessageBox)
from PyQt6.QtCore import Qt
from .worker import MiningWorker
from .mining import BATCH_SIZE, OPS_PER_HASH
from .utils import print_hex_le
from .network import RPCClient
from .wallet import generate_keypair, save_keys, load_wallet

def format_time_estimate(seconds):
    """Format estimated time in a human readable way."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensCoin Miner")
        self.worker = None
        self.coins_mined = 0
        self.rpc = None
        self.initUI()
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Input parameters section
        param_layout = QVBoxLayout()
        param_frame = QFrame()
        param_frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        param_frame.setLayout(param_layout)
        
        # Node connection
        node_layout = QHBoxLayout()
        node_label = QLabel("Node:")
        self.node_input = QLineEdit()
        self.node_input.setText("65.109.105.241:8332")  # Updated default port
        node_layout.addWidget(node_label)
        node_layout.addWidget(self.node_input)
        param_layout.addLayout(node_layout)
        
        # Mining address
        address_layout = QHBoxLayout()
        address_label = QLabel("Mining Address:")
        self.address_input = QLineEdit()
        self.generate_address_button = QPushButton("Generate New")
        self.generate_address_button.clicked.connect(self.generate_new_address)
        address_layout.addWidget(address_label)
        address_layout.addWidget(self.address_input)
        address_layout.addWidget(self.generate_address_button)
        param_layout.addLayout(address_layout)
        
        # Load any existing address
        wallet = load_wallet()
        if wallet:
            self.address_input.setText(wallet[-1]['address'])
        
        # Batch size input
        batch_layout = QHBoxLayout()
        batch_label = QLabel("Batch Size:")
        self.batch_input = QLineEdit()
        self.batch_input.setText(str(BATCH_SIZE))
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_input)
        param_layout.addLayout(batch_layout)
        
        # Device selection
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "mps", "cpu"])
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        param_layout.addLayout(device_layout)
        
        layout.addWidget(param_frame)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Mining")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_mining)
        self.stop_button.clicked.connect(self.stop_mining)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        
        # Stats section
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        stats_layout = QGridLayout()
        stats_frame.setLayout(stats_layout)
        
        # Create labels for stats
        self.elapsed_label = QLabel("Time: 0.0s")
        self.hashrate_label = QLabel("Hash Rate: 0 h/s")
        self.tops_label = QLabel("TOPS: 0.000000")
        self.total_label = QLabel("Total Hashes: 0")
        self.coins_label = QLabel("TensCoins Mined: 0")
        self.best_label = QLabel("Best: 0 bits")
        self.target_label = QLabel("Target: ? zeros")  # Shows required zeros
        self.estimate_label = QLabel("Est. Time: N/A")  # New estimate label
        
        # Add stats labels to grid
        stats_layout.addWidget(QLabel("Stats:"), 0, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        stats_layout.addWidget(self.elapsed_label, 1, 0)
        stats_layout.addWidget(self.hashrate_label, 1, 1)
        stats_layout.addWidget(self.tops_label, 2, 0)
        stats_layout.addWidget(self.total_label, 2, 1)
        stats_layout.addWidget(self.target_label, 3, 0)
        stats_layout.addWidget(self.best_label, 3, 1)
        stats_layout.addWidget(self.estimate_label, 4, 0)  # Added estimate
        stats_layout.addWidget(self.coins_label, 4, 1)
        
        layout.addWidget(stats_frame)
        
        # Status display
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)
        
        # Add status message for loaded address
        if wallet and len(wallet) > 0:
            self.status_text.append(f"Loaded saved address: {wallet[-1]['address']}")
        
        # Window setup
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        # Auto-detect best device
        if torch.cuda.is_available():
            self.device_combo.setCurrentText("cuda")
        elif torch.backends.mps.is_available():
            self.device_combo.setCurrentText("mps")
        else:
            self.device_combo.setCurrentText("cpu")
    
    def generate_new_address(self):
        """Generate a new TensCoin address locally."""
        try:
            # Generate new keypair
            private_key, address = generate_keypair()
            
            # Save to wallet file
            wallet_path = save_keys(private_key, address)
            
            # Update UI
            self.address_input.setText(address)
            self.status_text.append(f"Generated new address: {address}")
            self.status_text.append(f"Private key saved to: {wallet_path}")
            
            # Show warning about backing up private key
            QMessageBox.warning(
                self,
                "Backup Warning",
                f"Private key saved to {wallet_path}\n\n" \
                "Please back up this file! If you lose it, you will lose access to any mined coins."
            )
            
        except Exception as e:
            self.status_text.append(f"Failed to generate address: {e}")
    
    def start_mining(self):
        # Parse node address
        node_addr = self.node_input.text().strip()
        try:
            host, port = node_addr.split(":")
            port = int(port)
        except ValueError:
            self.status_text.append("Error: Invalid node address (use format host:port)")
            return
        
        # Validate mining address
        address = self.address_input.text().strip()
        if not address:
            self.status_text.append("Error: Mining address required")
            return
            
        try:
            batch_size = int(self.batch_input.text())
            if batch_size < 1:
                raise ValueError()
        except ValueError:
            self.status_text.append("Error: Invalid batch size")
            return
        
        device = self.device_combo.currentText()
        
        # Clear status and reset stats
        self.status_text.clear()
        self.status_text.append(f"Connecting to node {node_addr}...")
        
        self.elapsed_label.setText("Time: 0.0s")
        self.hashrate_label.setText("Hash Rate: 0 h/s")
        self.tops_label.setText("TOPS: 0.000000")
        self.total_label.setText("Total Hashes: 0")
        self.best_label.setText("Best: 0 bits")
        self.target_label.setText("Target: ? zeros")
        self.estimate_label.setText("Est. Time: N/A")
        
        # Initialize RPC client
        self.rpc = RPCClient(host, port)
        
        try:
            # Get initial mining info via RPC
            mining_info = self.rpc.get_mining_info()
            target = mining_info.get('target', 'unknown')
            target_bits = mining_info.get('target_bits', 0)
            self.target_label.setText(f"Target: {target_bits} zeros")
            self.status_text.append(f"Connected successfully. Need {target_bits} leading zeros")
            self.status_text.append(f"Target: {target}")
            self.status_text.append(f"Mining to address: {address}")
            print(f"Full mining info:", json.dumps(mining_info, indent=2))
        except Exception as e:
            self.status_text.append(f"Failed to connect to node: {e}")
            return
        
        # Disable/enable controls
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.node_input.setEnabled(False)
        self.batch_input.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.address_input.setEnabled(False)
        self.generate_address_button.setEnabled(False)
        
        # Start mining thread
        self.worker = MiningWorker(self.rpc, batch_size, device, address)
        self.worker.progress.connect(self.update_progress)
        self.worker.solution.connect(self.solution_found)
        self.worker.status.connect(self.status_update)
        self.worker.start()
    
    def stop_mining(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        
        # Re-enable controls
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.node_input.setEnabled(True)
        self.batch_input.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.address_input.setEnabled(True)
        self.generate_address_button.setEnabled(True)
        
        self.status_text.append("Mining stopped.")
    
    def update_estimate(self, hash_rate, target_bits):
        """Update the time estimate based on hash rate and target."""
        if hash_rate > 0:
            # Probability is 1/2^target_bits
            avg_hashes_needed = 2 ** target_bits
            seconds = avg_hashes_needed / hash_rate
            self.estimate_label.setText(f"Est. Time: {format_time_estimate(seconds)}")
        else:
            self.estimate_label.setText("Est. Time: N/A")
    
    def update_progress(self, data):
        elapsed = data['elapsed_time']
        attempts = data['attempts']
        best_bits = data['best_zero_bits']
        
        # Calculate hash rate and TOPS
        hash_rate = attempts / elapsed if elapsed > 0 else 0
        tops = (hash_rate * OPS_PER_HASH) / 1e12
        
        # Update stat labels
        self.elapsed_label.setText(f"Time: {elapsed:.1f}s")
        self.hashrate_label.setText(f"Hash Rate: {hash_rate:.1f} h/s")
        self.tops_label.setText(f"TOPS: {tops:.6f}")
        self.total_label.setText(f"Total Hashes: {attempts:,}")
        self.best_label.setText(f"Best: {best_bits} bits")
        
        # Update time estimate
        target_text = self.target_label.text()
        try:
            target_bits = int(target_text.split()[1])
        except:
            target_bits = 24
        self.update_estimate(hash_rate, target_bits)
    
    def status_update(self, message):
        """Handle status updates from worker."""
        self.status_text.append(message)
        
        # Scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def solution_found(self, data):
        self.coins_mined += 1
        self.coins_label.setText(f"TensCoins Mined: {self.coins_mined}")
        
        nonce = print_hex_le(data['nonce'])
        hash_str = print_hex_le(data['hash'])
        attempts = data['attempts']
        total_time = data['time']
        
        self.status_text.append("\nSolution found!")
        self.status_text.append(f"Nonce: {nonce}")
        self.status_text.append(f"Hash:  {hash_str}")
        
        # Update stats
        hash_rate = attempts / total_time
        tops = (hash_rate * OPS_PER_HASH) / 1e12
        
        self.elapsed_label.setText(f"Time: {total_time:.1f}s")
        self.hashrate_label.setText(f"Hash Rate: {hash_rate:.1f} h/s")
        self.total_label.setText(f"Total Hashes: {attempts:,}")
        self.tops_label.setText(f"TOPS: {tops:.6f}")
        
        try:
            # Get updated mining info after found block
            mining_info = self.rpc.get_mining_info()
            target = mining_info.get('target', 'unknown')
            target_bits = mining_info.get('target_bits', 0)
            self.target_label.setText(f"Target: {target_bits} zeros")
            self.status_text.append(f"Updated target: need {target_bits} leading zeros")
            self.status_text.append(f"Target: {target}")
        except Exception as e:
            self.status_text.append(f"Failed to update target: {e}")