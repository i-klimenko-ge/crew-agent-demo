from dataclasses import dataclass, field
from typing import List, Dict
import os

@dataclass
class Blackboard:
    """Simple in-memory blackboard for agent communication.

    In addition to keeping the notes in memory, all notes are also appended to
    a ``notes.txt`` file in the project root. The file is cleared when this
    module is first imported so each run starts with an empty log.
    """
    notes: List[Dict[str, str]] = field(default_factory=list)
    file_path: str = "notes.txt"

    def __post_init__(self):
        # Clear the notes file on startup
        with open(self.file_path, "w", encoding="utf-8"):
            pass

    def post(self, author: str, content: str) -> None:
        """Store a note from an agent and write it to ``notes.txt``."""
        self.notes.append({"author": author, "content": content})
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write("---------------------------------------------------------------------------\n")
            f.write(f"{author}:\n")
            f.write("---------------------------------------------------------------------------\n")
            f.write(f"{content}\n")
            f.write("---------------------------------------------------------------------------\n\n")

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of all notes."""
        return list(self.notes)

# Singleton instance used across agents
blackboard = Blackboard()
