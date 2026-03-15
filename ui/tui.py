"""
Terminal UI components using Rich library.

Provides visual interface for displaying agent responses.
"""
from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme


AGENT_THEME = Theme(
    {
        "info": "bright_cyan",
        "warning": "bold yellow",
        "error": "bright_red bold",
        "success": "bright_green bold",
        "dim": "grey66",
        "muted": "grey50",
        "border": "grey50",
        "highlight": "bold bright_cyan",
        "user": "bright_blue bold",
        "assistant": "bright_white italic",
        "tool": "bright_magenta bold",
        "tool.read": "bright_cyan",
        "tool.write": "bright_yellow",
        "tool.shell": "bright_magenta",
        "tool.network": "blue bold",
        "tool.memory": "bright_green",
        "tool.mcp": "cyan bold",
        "code": "white",
    }
)

_console: Console | None = None


def get_console() -> Console:
    """Get or create global console with AGENT_THEME."""
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)
    return _console


class TUI:
    """Terminal UI for displaying agent interactions."""
    
    def __init__(self, console: Console | None = None) -> None:
        """Initialize TUI with console (defaults to global)."""
        self.console = console or get_console()
        self._assistant_stream_open = False

    def begin_assistant(self) -> None:
        """Start assistant message section with header."""
        self.console.print()
        self.console.print(Rule(Text("Assistant", style="assistant")))
        self._assistant_stream_open = True
    
    def end_assistant(self) -> None:
        """End assistant section and reset stream flag."""
        if self._assistant_stream_open:
            self.console.print()
            self._assistant_stream_open = False
    
    def stream_assistant_delta(self, content: str) -> None:
        """Stream text chunk without markup."""
        self.console.print(content, end="", markup=False)
