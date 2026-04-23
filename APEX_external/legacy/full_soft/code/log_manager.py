"""
Centralized Logging System
Provides unified logging interface for the entire project
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname:8}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


class LogManager:
    """
    Centralized logging manager.
    Handles log file creation, formatting, and provides loggers for different components.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        enable_colors: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.console_level = getattr(logging, console_level.upper())
        self.file_level = getattr(logging, file_level.upper())
        self.enable_colors = enable_colors
        
        # Create timestamp-based log directory
        self.session_timestamp = datetime.now()
        self.date_str = self.session_timestamp.strftime("%Y-%m-%d")
        self.time_str = self.session_timestamp.strftime("%H-%M-%S")
        self.session_log_dir = self.log_dir / self.date_str / self.time_str
        
        # Create log directory
        self.session_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Track created loggers
        self._loggers = {}
        
        # Create main logger
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup root logger with console and file handlers"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        
        if self.enable_colors:
            console_format = ColoredFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            console_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
        
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
        
        # Main file handler
        main_log_file = self.session_log_dir / "main.log"
        file_handler = logging.FileHandler(main_log_file, mode='a')
        file_handler.setLevel(self.file_level)
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging initialized. Session directory: {self.session_log_dir}")
    
    def get_logger(
        self,
        name: str,
        create_file: bool = False,
        log_format: str = "standard"
    ) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (typically module or component name)
            create_file: If True, create a separate log file for this logger
            log_format: Format style - "standard" or "sensor" (minimal format for sensor data)
        
        Returns:
            Logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        
        # If separate file requested, add file handler
        if create_file:
            log_file = self.session_log_dir / f"{name}.log"
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            
            if log_format == "sensor":
                # Minimal format for sensor data
                formatter = logging.Formatter(
                    '%(relativeCreated).3f\t%(message)s'
                )
            else:
                formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        self._loggers[name] = logger
        return logger
    
    def backup_config(self, config_path: str = "config.json"):
        """Backup configuration file to log directory"""
        if os.path.exists(config_path):
            try:
                import shutil
                backup_path = self.session_log_dir / "config.json"
                shutil.copy(config_path, backup_path)
                logging.info(f"Configuration backed up to {backup_path}")
            except Exception as e:
                logging.error(f"Failed to backup configuration: {e}")
        else:
            logging.warning(f"Configuration file not found: {config_path}")
    
    def get_session_dir(self) -> Path:
        """Get current session log directory"""
        return self.session_log_dir
    
    def close_all(self):
        """Close all file handlers"""
        for logger in self._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


class ComponentLogger:
    """
    Component-specific logger wrapper.
    Provides convenient logging methods with component context.
    """
    
    def __init__(self, logger: logging.Logger, component_name: str):
        self._logger = logger
        self.component_name = component_name
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._logger.debug(f"[{self.component_name}] {message}", **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._logger.info(f"[{self.component_name}] {message}", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._logger.warning(f"[{self.component_name}] {message}", **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._logger.error(f"[{self.component_name}] {message}", **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._logger.critical(f"[{self.component_name}] {message}", **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self._logger.exception(f"[{self.component_name}] {message}", **kwargs)


# Global log manager instance
_log_manager: Optional[LogManager] = None


def initialize_logging(
    log_dir: str = "logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    enable_colors: bool = True,
    backup_config: bool = True
) -> LogManager:
    """
    Initialize the global logging system.
    
    Args:
        log_dir: Base directory for logs
        console_level: Logging level for console output
        file_level: Logging level for file output
        enable_colors: Enable colored console output
        backup_config: Backup config.json to log directory
    
    Returns:
        LogManager instance
    """
    global _log_manager
    
    if _log_manager is None:
        _log_manager = LogManager(
            log_dir=log_dir,
            console_level=console_level,
            file_level=file_level,
            enable_colors=enable_colors
        )
        
        if backup_config:
            _log_manager.backup_config()
    
    return _log_manager


def get_logger(name: str, create_file: bool = False, log_format: str = "standard") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        create_file: Create separate log file for this logger
        log_format: Format style - "standard" or "sensor"
    
    Returns:
        Logger instance
    """
    global _log_manager
    
    if _log_manager is None:
        initialize_logging()
    
    return _log_manager.get_logger(name, create_file, log_format)


def get_component_logger(component_name: str, create_file: bool = False) -> ComponentLogger:
    """
    Get a component-specific logger.
    
    Args:
        component_name: Name of the component
        create_file: Create separate log file for this component
    
    Returns:
        ComponentLogger instance
    """
    logger = get_logger(component_name, create_file)
    return ComponentLogger(logger, component_name)


def shutdown_logging():
    """Shutdown logging system and close all handlers"""
    global _log_manager
    
    if _log_manager is not None:
        _log_manager.close_all()
        _log_manager = None


if __name__ == "__main__":
    # Test the logging system
    print("=== Logging System Test ===\n")
    
    # Initialize logging
    log_mgr = initialize_logging(console_level="DEBUG")
    log_mgr.backup_config("config_backup.json")
    
    # Get different loggers
    main_logger = get_logger("Main")
    lidar_logger = get_logger("LiDAR", create_file=True, log_format="sensor")
    motor_logger = get_component_logger("Motor", create_file=True)
    
    # Test logging at different levels
    main_logger.debug("This is a debug message")
    main_logger.info("System initialized successfully")
    main_logger.warning("Low battery detected")
    main_logger.error("Failed to connect to sensor")
    
    
    # Test component logger
    motor_logger.info("Motor started")
    motor_logger.debug("Speed set to 1.5 m/s")
    
    # Test sensor logger (minimal format)
    lidar_logger.info("[0.5, 1.2, 0.8, ...]")
    
    # Simulate exception
    try:
        raise ValueError("Simulated error")
    except Exception:
        main_logger.exception("An error occurred")
    
    print(f"\nLog files created in: {log_mgr.get_session_dir()}")
    
    # Cleanup
    shutdown_logging()