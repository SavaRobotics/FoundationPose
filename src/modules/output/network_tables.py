#!/usr/bin/env python3

import os
import time
import sys
from networktables import NetworkTables
from dotenv import load_dotenv

load_dotenv()

# Import the NT schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from nt_schema import *
except ImportError:
    print("ERROR: Could not import nt_schema")
    sys.exit(1)

class NetworkTablesPublisher:
    def __init__(self, server_url=os.environ.get('SERVER_URL', 'localhost')):
        # Get NT server address from environment variable
        self.nt_server = server_url
        
        # Connect to NetworkTables
        NetworkTables.initialize(server=self.nt_server)
        
        # Wait for NT connection
        while not NetworkTables.isConnected():
            print("Waiting for NetworkTables connection...")
            time.sleep(1)
        
        print(f"Connected to NetworkTables server at {self.nt_server}")
        
        # Create table for publishing foundation pose
        self.vision_table = NetworkTables.getTable(VISION_TABLE)
    
    def publish_pose(self, pose_6d):
        """Publish a 6D pose to NetworkTables"""
        # Format as string "x,y,z,roll,pitch,yaw"
        pose_str = f"{pose_6d[0]:.3f},{pose_6d[1]:.3f},{pose_6d[2]:.3f},{pose_6d[3]:.3f},{pose_6d[4]:.3f},{pose_6d[5]:.3f}"
        
        # Publish to NT using constants from schema
        self.vision_table.putString(FOUNDATION_POSE.split('/')[-1], pose_str)
        
        # Update timestamp for latency tracking
        timestamp_key = VISION_TIMESTAMP.split('/')[-1]
        self.vision_table.putNumber(timestamp_key, time.time())
        
        print(f"Published pose: {pose_str}")
        
        return pose_str
    
    def process(self, data):
        """Process data object and publish pose"""
        if data.pose_6d is None:
            print("Error: No pose data available")
            return data
            
        self.publish_pose(data.pose_6d)
        return data