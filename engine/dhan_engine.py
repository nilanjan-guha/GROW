import json
import urllib.request
import traceback
from datetime import datetime

class DhanLiveEngine:
    """
    Dhan HQ Trading & Account Integration Engine
    Manages authenticated connection with Dhan API v2
    Provides real-time fund limit, profile verification, and trade position tracking.
    """
    def __init__(self, token=None, client_id="1100857857"):
        self.token = token or "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJ1c2VyUmVnaW9uIjoiUjEiLCJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzk1NDk1MDY0LCJpYXQiOjE3OTU0MDg2NjQsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwODU3ODU3In0.OWzwU9kZ1bt4iSsrCWt4bQQs004zskXyzeSIuxVqIRcg5uYvbHWyyrnepAv2ZzZrzoBPeD51feG1Kh0vjO8iUA"
        self.client_id = client_id
        self.base_url = "https://api.dhan.co/v2"
        self.is_connected = False
        self._test_connection()

    def _headers(self):
        return {
            "access-token": self.token,
            "client-id": self.client_id,
            "Content-Type": "application/json"
        }

    def _test_connection(self):
        try:
            url = f"{self.base_url}/fundlimit"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.getcode() == 200:
                    self.is_connected = True
                    print(f"[DHAN ENGINE] Successfully connected! Client ID: {self.client_id}")
        except Exception as e:
            self.is_connected = False
            print("[DHAN ENGINE] Connection test error:", e)

    def get_funds(self):
        """Returns available trading capital and utilized limits from Dhan"""
        if not self.is_connected:
            return {}
        try:
            url = f"{self.base_url}/fundlimit"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}

    def get_positions(self):
        """Returns open trading positions"""
        if not self.is_connected:
            return []
        try:
            url = f"{self.base_url}/positions"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return []

    def get_orders(self):
        """Returns active and executed orders for the session"""
        if not self.is_connected:
            return []
        try:
            url = f"{self.base_url}/orders"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return []
