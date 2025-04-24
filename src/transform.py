import numpy as np
import math

class PoseTransformer():
    """Transforms 4x4 pose matrix to 6D pose (x, y, z, roll, pitch, yaw)."""
    
    def __init__(self, to_inches=True, to_degrees=True):
        self.to_inches = to_inches
        self.to_degrees = to_degrees
    
    def transform_pose(self, center_pose):
        """Transform the pose matrix to a 6D pose."""
        # Convert the pose matrix to 6D pose
        pose_6d = self._convert_pose_matrix_to_6d(center_pose)
        
        # Print the result
        x, y, z, roll, pitch, yaw = pose_6d
        unit_pos = "inches" if self.to_inches else "meters"
        unit_rot = "degrees" if self.to_degrees else "radians"
        print(f"📏 Object position ({unit_pos}): x={x:.4f}, y={y:.4f}, z={z:.4f}")
        print(f"🔄 Object rotation ({unit_rot}): roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")
        
        return pose_6d

    def _convert_pose_matrix_to_6d(self, pose_matrix):
        """Convert 4x4 pose matrix to 6D pose (x, y, z, roll, pitch, yaw)."""
        # Extract translation
        x = pose_matrix[0, 3]
        y = pose_matrix[1, 3]
        z = pose_matrix[2, 3]
        
        # Extract rotation matrix and convert to Euler angles
        R = pose_matrix[:3, :3]
        roll, pitch, yaw = self._rotation_matrix_to_euler_angles(R)
        
        # Convert radians to degrees if requested
        if self.to_degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        
        # Convert meters to inches if requested (1 meter = 39.3701 inches)
        if self.to_inches:
            x = x * 39.3701
            y = y * 39.3701
            z = z * 39.3701
        
        return x, y, z, roll, pitch, yaw
        
    def _rotation_matrix_to_euler_angles(self, R):
        """Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in radians."""
        # Check if we're in the singularity known as "Gimbal lock"
        if abs(R[2, 0]) > 0.9999:
            # Special case: pitch is around ±90°
            yaw = 0
            if R[2, 0] < 0:
                pitch = math.pi/2
                roll = math.atan2(R[0, 1], R[1, 1])
            else:
                pitch = -math.pi/2
                roll = -math.atan2(R[0, 1], R[1, 1])
        else:
            # Standard case
            pitch = -math.asin(R[2, 0])
            roll = math.atan2(R[2, 1]/math.cos(pitch), R[2, 2]/math.cos(pitch))
            yaw = math.atan2(R[1, 0]/math.cos(pitch), R[0, 0]/math.cos(pitch))
        
        return roll, pitch, yaw