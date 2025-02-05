"""RPC client for TensCoin network."""

import json
import requests
from typing import Dict, Any, Optional

class RPCClient:
    """JSON-RPC client for interacting with TensCoin node."""
    
    def __init__(self, host: str = "65.109.105.241", port: int = 8332,
                 username: str = "tenscoin", password: str = "tenscoin"):
        """Initialize RPC client."""
        self.url = f"http://{host}:{port}"
        self.username = username
        self.password = password
        self._id = 0
        
    def _call(self, method: str, params: list = None) -> Dict[str, Any]:
        """Make RPC call."""
        self._id += 1
        headers = {'content-type': 'application/json'}
        payload = {
            "method": method,
            "params": params or [],
            "jsonrpc": "2.0",
            "id": self._id,
        }
        
        try:
            response = requests.post(
                self.url,
                auth=(self.username, self.password),
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            response.raise_for_status()
            resp_json = response.json()
            print(f"RPC Response for {method}:", json.dumps(resp_json, indent=2))
            if 'error' in resp_json:
                raise ConnectionError(f"RPC error: {resp_json['error']}")
            return resp_json.get('result')
        except Exception as e:
            raise ConnectionError(f"RPC call failed: {e}")

    def get_mining_info(self) -> Dict[str, Any]:
        """Get current mining-related information."""
        result = self._call('getmininginfo')
        if result and 'target' in result:
            # Count leading zeros in target (each '0' is 4 bits)
            target = result['target']
            leading_zeros = 0
            for c in target:
                if c == '0':
                    leading_zeros += 4
                else:
                    break
            result['target_bits'] = leading_zeros
        return result
        
    def get_block_template(self) -> Dict[str, Any]:
        """Get block template for mining."""
        return self._call('getblocktemplate', [{"rules": ["segwit"]}])
        
    def submit_block(self, block_hex: str) -> bool:
        """Submit solved block."""
        try:
            result = self._call('submitblock', [block_hex])
            # submitblock returns null on success
            return result is None
        except:
            return False