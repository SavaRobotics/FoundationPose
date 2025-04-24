# src/modules/output/network_tables.py
import time
from ...pipeline.base import Processor

class NetworkTablesPublisher(Processor):
    """Publishes pose data to NetworkTables."""
    
    def __init__(self, server_url=None, vision_table="Vision", entry_name="FoundationPose", timeout=10):
        self.server_url = server_url
        self.vision_table = vision_table
        self.entry_name = entry_name
        self.timeout = timeout
        self.nt = None
        self.nt_available = False
        
        # Check if NetworkTables is available
        try:
            from networktables import NetworkTables
            self.NetworkTables = NetworkTables
            self.nt_available = True
        except ImportError:
            print("⚠️ Warning: Cannot import NetworkTables. Please install with: pip install pynetworktables")
            self.nt_available = False
    
    def process(self, data):
        """Publish pose data to NetworkTables."""
        if not self.nt_available:
            data.add_error("NetworkTablesPublisher", "NetworkTables is not available")
            return data
            
        if data.pose_6d is None:
            data.add_error("NetworkTablesPublisher", "No pose data available")
            return data
            
        # Use the URL from data if not provided in constructor
        server_url = self.server_url
        if server_url is None and data.base_url is not None:
            from urllib.parse import urlparse
            parsed_url = urlparse(data.base_url)
            server_url = parsed_url.netloc
            if not server_url:
                server_url = data.base_url.replace("http://", "").replace("https://", "").split("/")[0]
        
        if not server_url:
            data.add_error("NetworkTablesPublisher", "No server URL available")
            return data
            
        print(f"📤 Publishing pose to NetworkTables at {server_url}, table {self.vision_table}, entry {self.entry_name}")
        
        try:
            # Connect to NetworkTables if not connected
            if self.nt is None:
                self.NetworkTables.initialize(server=server_url)
                # Wait for connection
                start_time = time.time()
                while not self.NetworkTables.isConnected() and (time.time() - start_time) < self.timeout:
                    print("Waiting for NetworkTables connection...")
                    time.sleep(0.5)
                
                if not self.NetworkTables.isConnected():
                    data.add_error("NetworkTablesPublisher", f"Failed to connect to NetworkTables server at {server_url}")
                    return data
                    
                print(f"✅ Connected to NetworkTables server at {server_url}")
                
            # Get the table and set the value
            table = self.NetworkTables.getTable(self.vision_table)
            
            # Format pose as comma-separated string
            pose_str = ",".join(map(str, data.pose_6d))
            table.putString(self.entry_name, pose_str)
            
            print(f"✅ Published pose to NetworkTables: {pose_str}")
            
        except Exception as e:
            data.add_error("NetworkTablesPublisher", f"NetworkTables error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return data
    

    