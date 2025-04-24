#!/usr/bin/env python3
"""
Utility to test camera intrinsics loading.

This script can be used to verify that camera intrinsics files can be properly loaded
by the FoundationPose pipeline.

Usage:
    python -m src.utils.tests.test_cam_intrinsics path/to/intrinsics/file.txt [camera_section]

Examples:
    python -m src.utils.tests.test_cam_intrinsics data/intrinsics/cam_K.txt
    python -m src.utils.tests.test_cam_intrinsics data/intrinsics/cam_K.txt LEFT_CAM_SVGA 
"""

import os
import sys
import re
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CameraIntrinsicsTest")

def load_camera_intrinsics(intrinsics_file, camera_section="LEFT_CAM_FHD1200"):
    """Load camera intrinsics from a file"""
    try:
        logger.info(f"Loading camera intrinsics from {intrinsics_file}")
        
        # First, check if the file exists
        if not os.path.exists(intrinsics_file):
            logger.error(f"❌ Camera intrinsics file not found: {intrinsics_file}")
            return None
            
        # Check if the file is in the parameter-based format
        with open(intrinsics_file, 'r') as f:
            first_line = f.readline().strip()
            logger.debug(f"First line of intrinsics file: '{first_line}'")
            
            if first_line.startswith('['):
                logger.info(f"Detected parameter-based format, using section: {camera_section}")
                return convert_camera_intrinsics(intrinsics_file, camera_section)
        
        # Original format (3x3 matrix)
        logger.info("Using matrix format for camera intrinsics")
        with open(intrinsics_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 3:
                K = np.array([
                    [float(val) for val in lines[0].strip().split()],
                    [float(val) for val in lines[1].strip().split()],
                    [float(val) for val in lines[2].strip().split()]
                ])
                logger.info(f"Read 3x3 camera matrix:\n{K}")
                return K
            else:
                logger.error(f"❌ Camera intrinsics file has too few lines: {len(lines)}")
                raise ValueError("Intrinsics file has incorrect format - needs at least 3 lines")
    except Exception as e:
        logger.error(f"❌ Error loading camera intrinsics: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def convert_camera_intrinsics(intrinsics_file, camera_section="LEFT_CAM_FHD1200"):
    """
    Convert camera intrinsics from parameter-based format to 3x3 matrix format
    
    Args:
        intrinsics_file: Path to the cam_K.txt file with parameter-based format
        camera_section: Camera section to use (e.g., "LEFT_CAM_FHD1200")
        
    Returns:
        3x3 numpy array with camera intrinsics matrix
    """
    try:
        logger.info(f"🔄 Converting camera intrinsics from {camera_section} section...")
        with open(intrinsics_file, "r") as f:
            content = f.read()

        # Show all available sections
        available_sections = re.findall(r"\[(.*?)\]", content)
        logger.info(f"Available sections: {available_sections}")

        # Find the section
        section_pattern = r"\[" + camera_section + r"\](.*?)(?=\[|$)"
        section_match = re.search(section_pattern, content, re.DOTALL)

        if section_match:
            section_text = section_match.group(1)
            logger.debug(f"Found section text: {section_text}")
            
            # Extract parameters
            try:
                fx = float(re.search(r"fx=([\d\.e-]+)", section_text).group(1))
                fy = float(re.search(r"fy=([\d\.e-]+)", section_text).group(1))
                cx = float(re.search(r"cx=([\d\.e-]+)", section_text).group(1))
                cy = float(re.search(r"cy=([\d\.e-]+)", section_text).group(1))
            except AttributeError as e:
                logger.error(f"❌ Failed to extract camera parameters: {e}")
                logger.error(f"Section text: {section_text}")
                raise ValueError(f"Missing parameters in {camera_section} section")
            
            # Create the 3x3 intrinsics matrix
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            logger.info(f"✅ Camera intrinsics converted successfully:")
            logger.info(f"   fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            logger.info(f"Camera matrix K:\n{K}")
            
            return K
        else:
            logger.error(f"❌ Camera section '{camera_section}' not found")
            raise ValueError(f"Camera section {camera_section} not found in intrinsics file")
    
    except Exception as e:
        logger.error(f"❌ Error converting camera intrinsics: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def list_camera_sections(intrinsics_file):
    """List all camera sections in the intrinsics file"""
    try:
        with open(intrinsics_file, "r") as f:
            content = f.read()
            
        # Find all sections
        sections = re.findall(r"\[(.*?)\]", content)
        
        logger.info(f"Found {len(sections)} camera sections in {intrinsics_file}:")
        for i, section in enumerate(sections):
            logger.info(f"  {i+1}. [{section}]")
            
        return sections
    except Exception as e:
        logger.error(f"❌ Error listing camera sections: {str(e)}")
        return []

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} path/to/intrinsics/file.txt [camera_section]")
        sys.exit(1)
        
    intrinsics_file = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(intrinsics_file):
        logger.error(f"❌ File not found: {intrinsics_file}")
        sys.exit(1)
        
    # List camera sections
    sections = list_camera_sections(intrinsics_file)
    
    # Get camera section from command line or use default
    if len(sys.argv) > 2:
        camera_section = sys.argv[2]
    else:
        camera_section = "LEFT_CAM_FHD1200"
        logger.info(f"Using default camera section: {camera_section}")
        
    # Load camera intrinsics
    K = load_camera_intrinsics(intrinsics_file, camera_section)
    
    if K is not None:
        logger.info("✅ Successfully loaded camera intrinsics!")
        print("\nCamera Matrix K:")
        print(K)
    else:
        logger.error("❌ Failed to load camera intrinsics")
        sys.exit(1)

if __name__ == "__main__":
    main() 