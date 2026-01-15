from utils import create_logger
from typing import TypeAlias, Literal
ScalingFactor: TypeAlias = Literal[2, 4, 8]


logger = create_logger(log_level="INFO", log_file_name="SwinIR")
