# NetworkTables schema for communication between ZED Box and other systems

# Root table name
ROOT_TABLE = "SavaRobot"

# Tables for different subsystems
COMMANDS_TABLE = f"{ROOT_TABLE}/Commands"
STATUS_TABLE = f"{ROOT_TABLE}/Status"
VISION_TABLE = f"{ROOT_TABLE}/Vision"
DIAGNOSTICS_TABLE = f"{ROOT_TABLE}/Diagnostics"

# Foundation Pose data (from computer vision)
FOUNDATION_POSE = f"{VISION_TABLE}/FoundationPose"  # 6D pose (x,y,z,roll,pitch,yaw) as comma-separated string

# Arm command entries
ARM_TARGET_POSITION = f"{COMMANDS_TABLE}/ArmTargetPosition"  # Target position as string "x,y,z" in inches
ARM_COMMAND_READY = f"{COMMANDS_TABLE}/ArmCommandReady"  # Boolean flag indicating a new command is ready

# Arm status entries
ARM_CURRENT_POSITION = f"{STATUS_TABLE}/ArmCurrentPosition"  # Current position as string "x,y,z" in inches
ARM_STATE = f"{STATUS_TABLE}/ArmState"  # Current state of the arm (idle, moving, error)
ARM_ERROR = f"{STATUS_TABLE}/ArmError"  # Error message if any
ARM_COMMAND_RECEIVED = f"{STATUS_TABLE}/ArmCommandReceived"  # Boolean flag indicating command was received
ARM_COMMAND_EXECUTED = f"{STATUS_TABLE}/ArmCommandExecuted"  # Boolean flag indicating command was executed

# Timestamps for latency tracking
COMMAND_TIMESTAMP = f"{COMMANDS_TABLE}/Timestamp"  # Timestamp when command was sent
VISION_TIMESTAMP = f"{VISION_TABLE}/Timestamp"  # Timestamp when vision data was updated

# NetworkTables Constants
NT_UPDATE_FREQUENCY = 50.0  # Hz
EXPECTED_LATENCY_MS = 20.0  # Maximum expected latency
