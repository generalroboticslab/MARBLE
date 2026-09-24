"""Simple file logger that duplicates stdout/stderr to a log file.

Uses Python's built-in buffering for efficiency - no threading overhead.
"""

import sys
from pathlib import Path


class TeeStream:
    """Writes to both original stream and a file, with efficient buffering.

    Python's file buffering (default 8KB for text) handles write efficiency.
    flush() is called at configurable intervals by the training loop.
    """

    def __init__(self, file_path: Path, original_stream, buffer_size: int = 8192):
        """Initialize tee stream.

        Args:
            file_path: Path to log file
            original_stream: Original sys.stdout or sys.stderr
            buffer_size: Buffer size in bytes (default 8KB)
        """
        self.file = open(file_path, 'w', buffering=buffer_size)
        self.original = original_stream
        self.closed = False

    def write(self, data: str):
        """Write to both original stream and file."""
        if self.closed:
            return
        self.original.write(data)
        self.file.write(data)

    def flush(self):
        """Flush both streams."""
        if self.closed:
            return
        self.original.flush()
        self.file.flush()

    def close(self):
        """Close file stream."""
        if not self.closed:
            self.file.close()
            self.closed = True

    def isatty(self) -> bool:
        # TeeStream writes to a file; report non-TTY so tools like tqdm disable
        # themselves and don't pollute the log with carriage-return escape sequences.
        return False

    def __getattr__(self, name):
        """Delegate other attributes to original stream."""
        return getattr(self.original, name)


class FileLogger:
    """Context manager for logging stdout/stderr to a file.

    Simple, efficient implementation using Python's built-in buffering.
    No threading, no queue - just duplicate writes with smart buffering.

    Usage:
        with FileLogger(log_path, flush_interval=10) as logger:
            for i in range(100):
                print(f"Iteration {i}")
                if i % 10 == 0:
                    logger.flush()
    """

    def __init__(self, log_path: Path, buffer_size: int = 8192):
        """Initialize file logger.

        Args:
            log_path: Path to log file
            buffer_size: Buffer size in bytes (default 8KB)
        """
        self.log_path = Path(log_path)
        self.buffer_size = buffer_size
        self.stdout_tee = None
        self.stderr_tee = None
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        """Start logging to file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        self.stdout_tee = TeeStream(self.log_path, self.original_stdout, self.buffer_size)
        self.stderr_tee = TeeStream(self.log_path, self.original_stderr, self.buffer_size)

        sys.stdout = self.stdout_tee
        sys.stderr = self.stderr_tee

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop logging and restore original streams."""
        if self.stdout_tee:
            self.stdout_tee.flush()
            self.stdout_tee.close()
        if self.stderr_tee:
            self.stderr_tee.flush()
            self.stderr_tee.close()

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def flush(self):
        """Manually flush buffers (call periodically during training)."""
        if self.stdout_tee:
            self.stdout_tee.flush()
        if self.stderr_tee:
            self.stderr_tee.flush()
