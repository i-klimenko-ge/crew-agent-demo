from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Blackboard:
    """Simple in-memory blackboard for agent communication."""
    notes: List[Dict[str, str]] = field(default_factory=list)

    def post(self, author: str, content: str) -> None:
        """Store a note from an agent."""
        self.notes.append({"author": author, "content": content})

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of all notes."""
        return list(self.notes)

# Singleton instance used across agents
blackboard = Blackboard()
