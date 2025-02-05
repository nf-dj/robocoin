"""RPC client implementation for TensHash miner."""

import json
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse

class RPCClient:
    """JSON-RPC client for interacting with TensCoin node."""
    
    def __init__(self, url: str = "http://65.109.105.241:8332",
                 username: str = "tenscoin",
                 password: str = "tenscoin"):
        """Initialize RPC client.
        
        Args:
            url: Full URL to RPC endpoint
            username: RPC auth username
            password: RPC auth password
        """
        self.url = url
        self.username = username
        self.password = password
        self._id = 0
        
    def _call(self, method: str, params: list = None) -> Dict[str, Any]:
        """Make RPC call.
        
        Args:
            method: RPC method name
            params: List of parameters for the call
            
        Returns:
            Dict containing the RPC response
            
        Raises:
            requests.exceptions.RequestException: If the RPC call fails
        """
        self._id += 1
        headers = {'content-type': 'application/json'}
        payload = {
            "method": method,
            "params": params or [],
            "jsonrpc": "2.0",
            "id": self._id,
        }
        
        response = requests.post(
            self.url,
            auth=(self.username, self.password),
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_mining_info(self) -> Dict[str, Any]:
        """Get current mining-related information.
        
        Returns:
            Dict containing difficulty, network hashrate, etc.
        """
        return self._call('getmininginfo')
    
    def get_block_template(self) -> Dict[str, Any]:
        """Get block template for mining.
        
        Returns:
            Dict containing block data to mine on
        """
        return self._call('getblocktemplate', [{"rules": ["segwit"]}])
    
    def submit_block(self, block_hex: str) -> Optional[str]:
        """Submit a mined block to the network.
        
        Args:
            block_hex: The hex-encoded block data
            
        Returns:
            None on success, error message on failure
        """
        return self._call('submitblock', [block_hex])
    
    def get_network_hashps(self) -> float:
        """Get the estimated network hashes per second.
        
        Returns:
            Estimated network hashes per second
        """
        return self._call('getnetworkhashps')
