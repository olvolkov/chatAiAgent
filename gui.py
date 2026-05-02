# -*- coding: utf-8 -*-
"""
GUI Application for AI Agent with Markdown support.
Provides a user-friendly interface with formatted markdown rendering.
"""

import json
import logging
import os
import re
import threading
import tkinter as tk
import webbrowser
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

from agent import AIAgent

# Configure logging for GUI
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gui.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration file read/write operations."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._default_config = {
            "routerapi_base_url": "https://routerai.ru/api/v1",
            "routerapi_api_key": "",
            "models": ["qwen/qwen3.6-flash"],
            "active_model": "qwen/qwen3.6-flash",
            "tavily_api_key": "",
            "tavily_max_results": 5,
            "memory_limit": 10
        }
        logger.debug(f"ConfigManager initialized with config file: {config_file}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file, or return defaults if file doesn't exist."""
        logger.debug(f"Loading configuration from {self.config_file}")
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.debug(f"Configuration loaded successfully: {list(config.keys())}")
                    return config
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load configuration: {e}", exc_info=True)
                return self._default_config.copy()
        logger.debug("Config file does not exist, using defaults")
        return self._default_config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        logger.debug(f"Saving configuration to {self.config_file}")
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.debug("Configuration saved successfully")
            return True
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}", exc_info=True)
            return False


class MarkdownToTkinter:
    """Converts Markdown to tkinter-compatible formatting."""

    def __init__(self, text_widget: scrolledtext.ScrolledText):
        self.text_widget = text_widget
        # Map from unique tag name -> URL for clickable links
        self._link_urls: dict = {}
        self._link_counter: int = 0
        logger.debug("MarkdownToTkinter initialized")

        # Configure tags for different markdown elements
        self._configure_tags()

    def _configure_tags(self):
        """Configure text tags for markdown elements."""
        # Headers
        self.text_widget.tag_config('h1', font=('Segoe UI', 16, 'bold'),
                                    foreground='#1a1a1a', spacing1=10, spacing3=10)
        self.text_widget.tag_config('h2', font=('Segoe UI', 14, 'bold'),
                                    foreground='#2a2a2a', spacing1=8, spacing3=8)
        self.text_widget.tag_config('h3', font=('Segoe UI', 12, 'bold'),
                                    foreground='#3a3a3a', spacing1=6, spacing3=6)
        self.text_widget.tag_config('h4', font=('Segoe UI', 11, 'bold'),
                                    foreground='#4a4a4a')
        self.text_widget.tag_config('h5', font=('Segoe UI', 10, 'bold'),
                                    foreground='#5a5a5a')
        self.text_widget.tag_config('h6', font=('Segoe UI', 9, 'bold'),
                                    foreground='#6a6a6a')

        # Text formatting
        self.text_widget.tag_config('bold', font=('Segoe UI', 10, 'bold'))
        self.text_widget.tag_config('italic', font=('Segoe UI', 10, 'italic'))
        self.text_widget.tag_config('bold_italic', font=('Segoe UI', 10, 'bold italic'))
        self.text_widget.tag_config('underline', underline=True)
        self.text_widget.tag_config('strikethrough', overstrike=True)

        # Code blocks
        self.text_widget.tag_config('code_block', font=('Consolas', 9),
                                    background='#f5f5f5', foreground='#d14',
                                    spacing1=5, spacing3=5, lmargin1=10, lmargin2=10)
        self.text_widget.tag_config('inline_code', font=('Consolas', 9),
                                    background='#f0f0f0', foreground='#d14')

        # Lists — levels 1..4
        for lvl in range(1, 5):
            margin = lvl * 20
            self.text_widget.tag_config(
                f'list_item_{lvl}', lmargin1=margin, lmargin2=margin + 20,
                font=('Segoe UI', 10))
            self.text_widget.tag_config(
                f'list_item_ordered_{lvl}', lmargin1=margin, lmargin2=margin + 20,
                font=('Segoe UI', 10))

        # Blockquotes — levels 1..4
        for lvl in range(1, 5):
            margin = lvl * 20
            self.text_widget.tag_config(
                f'blockquote_{lvl}',
                lmargin1=margin, lmargin2=margin,
                foreground='#555555',
                background='#f9f9f9',
                font=('Segoe UI', 10, 'italic'))

        # Links base tag (visual style); per-link tags are created dynamically
        self.text_widget.tag_config('link', foreground='#0066cc', underline=True)

        # Horizontal rule
        self.text_widget.tag_config('hr', font=('Consolas', 8),
                                    foreground='#aaaaaa', spacing1=6, spacing3=6)

        # Table tags
        self.text_widget.tag_config('table_header', font=('Consolas', 9, 'bold'),
                                    background='#e8e8e8', foreground='#000000')
        self.text_widget.tag_config('table_row', font=('Consolas', 9),
                                    background='#ffffff', foreground='#000000')
        self.text_widget.tag_config('table_row_alt', font=('Consolas', 9),
                                    background='#f5f5f5', foreground='#000000')

        # Normal text
        self.text_widget.tag_config('normal', font=('Segoe UI', 10))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_markdown(self, markdown_text: str):
        """
        Parse *markdown_text* and insert it directly into the text widget
        at the current insertion point.  The widget must be in 'normal' state.
        """
        logger.debug(f"Inserting markdown text (length: {len(markdown_text)} chars)")
        lines = markdown_text.split('\n')
        logger.debug(f"Split markdown into {len(lines)} lines")
        in_code_block = False
        code_lines: list = []
        in_table = False
        table_lines: list = []
        # ordered-list counters per indent level
        ordered_counters: dict = {}

        i = 0
        while i < len(lines):
            line = lines[i]

            # ── fenced code block ──────────────────────────────────────
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_lines = []
                else:
                    self._insert_code_block('\n'.join(code_lines))
                    in_code_block = False
                    code_lines = []
                i += 1
                continue

            if in_code_block:
                code_lines.append(line)
                i += 1
                continue

            # ── table detection ────────────────────────────────────────
            if '|' in line:
                # peek ahead: if next line is a separator row, it's a table
                if not in_table:
                    if i + 1 < len(lines) and re.match(r'^\s*\|?[\s\-:]+\|', lines[i + 1]):
                        in_table = True
                        table_lines = [line]
                        i += 1
                        continue
                if in_table:
                    # separator row — skip
                    if re.match(r'^\s*\|?[\s\-:|]+\|', line):
                        i += 1
                        continue
                    table_lines.append(line)
                    # check if table ends
                    if i + 1 >= len(lines) or '|' not in lines[i + 1]:
                        self._insert_table(table_lines)
                        in_table = False
                        table_lines = []
                    i += 1
                    continue

            # flush table if we left it
            if in_table:
                self._insert_table(table_lines)
                in_table = False
                table_lines = []

            # ── process single line ────────────────────────────────────
            self._process_line(line, ordered_counters)
            i += 1

        # flush leftovers
        if in_code_block and code_lines:
            self._insert_code_block('\n'.join(code_lines))
        if in_table and table_lines:
            self._insert_table(table_lines)

    # ------------------------------------------------------------------
    # Line-level processors
    # ------------------------------------------------------------------

    def _process_line(self, line: str, ordered_counters: dict):
        """Insert one line into the widget with appropriate tags."""
        tw = self.text_widget

        # ── headers ───────────────────────────────────────────────────
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            tag = f'h{level}'
            self._insert_inline(m.group(2), base_tag=tag)
            tw.insert(tk.END, '\n', tag)
            return

        # ── horizontal rule ───────────────────────────────────────────
        if re.match(r'^\s*([-*_])\s*(\1\s*){2,}$', line):
            tw.insert(tk.END, '─' * 60 + '\n', 'hr')
            return

        # ── blockquote (supports nesting: >, >>, >>> …) ───────────────
        bq_match = re.match(r'^(>+)\s?(.*)', line)
        if bq_match:
            depth = min(len(bq_match.group(1)), 4)
            tag = f'blockquote_{depth}'
            prefix = '│ ' * depth
            tw.insert(tk.END, prefix, tag)
            self._insert_inline(bq_match.group(2), base_tag=tag)
            tw.insert(tk.END, '\n', tag)
            return

        # ── unordered list (supports nesting via leading spaces) ──────
        ul_match = re.match(r'^(\s*)[-*+]\s+(.*)', line)
        if ul_match:
            indent = len(ul_match.group(1))
            level = min(indent // 2 + 1, 4)
            tag = f'list_item_{level}'
            bullet = ('•', '◦', '▸', '▹')[level - 1]
            tw.insert(tk.END, '  ' * (level - 1) + bullet + ' ', tag)
            self._insert_inline(ul_match.group(2), base_tag=tag)
            tw.insert(tk.END, '\n', tag)
            return

        # ── ordered list (supports nesting) ───────────────────────────
        ol_match = re.match(r'^(\s*)(\d+)\.\s+(.*)', line)
        if ol_match:
            indent = len(ol_match.group(1))
            level = min(indent // 2 + 1, 4)
            tag = f'list_item_ordered_{level}'
            # increment counter for this level; reset deeper levels
            ordered_counters[level] = ordered_counters.get(level, 0) + 1
            for deeper in list(ordered_counters.keys()):
                if deeper > level:
                    del ordered_counters[deeper]
            number = ordered_counters[level]
            tw.insert(tk.END, '  ' * (level - 1) + f'{number}. ', tag)
            self._insert_inline(ol_match.group(3), base_tag=tag)
            tw.insert(tk.END, '\n', tag)
            return

        # ── empty line ────────────────────────────────────────────────
        if not line.strip():
            tw.insert(tk.END, '\n', 'normal')
            return

        # ── regular paragraph text ────────────────────────────────────
        self._insert_inline(line, base_tag='normal')
        tw.insert(tk.END, '\n', 'normal')

    # ------------------------------------------------------------------
    # Inline formatting
    # ------------------------------------------------------------------

    def _insert_inline(self, text: str, base_tag: str = 'normal'):
        """
        Parse inline markdown in *text* and insert into the widget.
        Handles: bold+italic, bold, italic, strikethrough, inline code, links.
        """
        # Tokenise the text into segments: (content, kind)
        # kinds: 'text', 'code', 'link'
        segments = self._tokenise_inline(text)
        for content, kind, extra in segments:
            if kind == 'code':
                self.text_widget.insert(tk.END, content, ('inline_code',))
            elif kind == 'link':
                url = extra
                self._insert_link(content, url, base_tag)
            elif kind == 'text':
                self._insert_styled_text(content, base_tag)

    def _tokenise_inline(self, text: str) -> list:
        """
        Split *text* into a list of (content, kind, extra) tuples.
        kind ∈ {'text', 'code', 'link'}
        extra = URL for links, '' otherwise.
        """
        result = []
        # Combined pattern: inline code, links, or plain text chunks
        pattern = re.compile(
            r'`(.+?)`'                          # inline code
            r'|\[([^\]]+)\]\(([^)]+)\)'         # [text](url)
            r'|([^`\[]+|\[(?!\w))'              # plain text
        )
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                result.append((text[pos:m.start()], 'text', ''))
            if m.group(1) is not None:
                result.append((m.group(1), 'code', ''))
            elif m.group(2) is not None:
                result.append((m.group(2), 'link', m.group(3)))
            elif m.group(4) is not None:
                result.append((m.group(4), 'text', ''))
            pos = m.end()
        if pos < len(text):
            result.append((text[pos:], 'text', ''))
        return result

    def _insert_styled_text(self, text: str, base_tag: str):
        """Insert *text* applying bold/italic/strikethrough on top of *base_tag*."""
        tw = self.text_widget
        # Pattern order matters: bold+italic before bold/italic individually
        pattern = re.compile(
            r'\*\*\*(.+?)\*\*\*'           # ***bold italic***
            r'|___(.+?)___'                 # ___bold italic___
            r'|\*\*(.+?)\*\*'              # **bold**
            r'|__(.+?)__'                  # __bold__
            r'|\*(.+?)\*'                  # *italic*
            r'|(?<!\w)_(.+?)_(?!\w)'       # _italic_
            r'|~~(.+?)~~'                  # ~~strikethrough~~
        )
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                tw.insert(tk.END, text[pos:m.start()], (base_tag,))
            if m.group(1) is not None:   # ***bold italic***
                tw.insert(tk.END, m.group(1), (base_tag, 'bold_italic'))
            elif m.group(2) is not None: # ___bold italic___
                tw.insert(tk.END, m.group(2), (base_tag, 'bold_italic'))
            elif m.group(3) is not None: # **bold**
                tw.insert(tk.END, m.group(3), (base_tag, 'bold'))
            elif m.group(4) is not None: # __bold__
                tw.insert(tk.END, m.group(4), (base_tag, 'bold'))
            elif m.group(5) is not None: # *italic*
                tw.insert(tk.END, m.group(5), (base_tag, 'italic'))
            elif m.group(6) is not None: # _italic_
                tw.insert(tk.END, m.group(6), (base_tag, 'italic'))
            elif m.group(7) is not None: # ~~strikethrough~~
                tw.insert(tk.END, m.group(7), (base_tag, 'strikethrough'))
            pos = m.end()
        if pos < len(text):
            tw.insert(tk.END, text[pos:], (base_tag,))

    def _insert_link(self, display_text: str, url: str, base_tag: str):
        """Insert a clickable hyperlink."""
        tw = self.text_widget
        self._link_counter += 1
        tag_name = f'link_{self._link_counter}'
        self._link_urls[tag_name] = url

        tw.tag_config(tag_name, foreground='#0066cc', underline=True)
        tw.tag_bind(tag_name, '<Button-1>',
                    lambda e, u=url: webbrowser.open(u))
        tw.tag_bind(tag_name, '<Enter>',
                    lambda e: tw.config(cursor='hand2'))
        tw.tag_bind(tag_name, '<Leave>',
                    lambda e: tw.config(cursor=''))

        tw.insert(tk.END, display_text, (base_tag, tag_name))

    # ------------------------------------------------------------------
    # Block-level helpers
    # ------------------------------------------------------------------

    def _insert_code_block(self, code: str):
        """Insert a fenced code block."""
        tw = self.text_widget
        tw.insert(tk.END, code + '\n', ('code_block',))

    def _insert_table(self, lines: list):
        """Render a Markdown table as fixed-width text."""
        tw = self.text_widget
        if not lines:
            return

        # Parse rows
        def parse_row(row_line):
            row_line = row_line.strip().strip('|')
            return [cell.strip() for cell in row_line.split('|')]

        rows = [parse_row(l) for l in lines]
        if not rows:
            return

        # Compute column widths
        col_count = max(len(r) for r in rows)
        col_widths = [0] * col_count
        for row in rows:
            for ci, cell in enumerate(row):
                if ci < col_count:
                    col_widths[ci] = max(col_widths[ci], len(cell))

        def fmt_row(row, widths):
            cells = []
            for ci, w in enumerate(widths):
                cell = row[ci] if ci < len(row) else ''
                cells.append(cell.ljust(w))
            return '│ ' + ' │ '.join(cells) + ' │'

        sep = '├─' + '─┼─'.join('─' * w for w in col_widths) + '─┤'
        top = '┌─' + '─┬─'.join('─' * w for w in col_widths) + '─┐'
        bot = '└─' + '─┴─'.join('─' * w for w in col_widths) + '─┘'

        tw.insert(tk.END, top + '\n', ('table_header',))
        for ri, row in enumerate(rows):
            tag = 'table_header' if ri == 0 else ('table_row' if ri % 2 == 1 else 'table_row_alt')
            tw.insert(tk.END, fmt_row(row, col_widths) + '\n', (tag,))
            if ri == 0:
                tw.insert(tk.END, sep + '\n', ('table_header',))
        tw.insert(tk.END, bot + '\n', ('table_row',))


class ChatMessage:
    """Represents a chat message."""
    
    def __init__(self, sender: str, content: str, message_type: str = 'text'):
        self.sender = sender
        self.content = content
        self.message_type = message_type  # 'text', 'error', 'system'


class SettingsDialog:
    """Dialog for editing configuration settings."""
    
    def __init__(self, parent: tk.Tk, config: Dict[str, Any]):
        self.parent = parent
        self.config = config.copy()
        self.result = None
        logger.debug(f"SettingsDialog initialized with config keys: {list(config.keys())}")
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("500x450")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center window
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (450 // 2)
        self.dialog.geometry(f"500x450+{x}+{y}")
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar for settings
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.settings_frame = ttk.Frame(canvas)
        
        self.settings_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Create input fields for each config option
        self._create_field("RouterAPI Base URL", "routerapi_base_url", 0)
        self._create_field("RouterAPI API Key", "routerapi_api_key", 1)
        
        # Models list (multi-line text area)
        ttk.Label(self.settings_frame, text="Models (one per line)", font=('Segoe UI', 9, 'bold')).grid(
            row=2, column=0, sticky=tk.W, pady=(5, 0))
        self._models_text = tk.Text(self.settings_frame, width=50, height=6)
        models = self.config.get("models", ["qwen/qwen3.6-flash"])
        self._models_text.insert('1.0', '\n'.join(models))
        self._models_text.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        
        # Active model dropdown
        ttk.Label(self.settings_frame, text="Active Model", font=('Segoe UI', 9, 'bold')).grid(
            row=3, column=0, sticky=tk.W, pady=(5, 0))
        self._active_model_var = tk.StringVar()
        self._active_model_combo = ttk.Combobox(
            self.settings_frame, 
            textvariable=self._active_model_var,
            width=47
        )
        models_list = self.config.get("models", ["qwen/qwen3.6-flash"])
        self._active_model_combo['values'] = models_list
        self._active_model_combo.set(self.config.get("active_model", models_list[0] if models_list else ""))
        self._active_model_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        
        self._create_field("Tavily API Key", "tavily_api_key", 4)
        self._create_field("Tavily Max Results", "tavily_max_results", 5)
        self._create_field("Memory Limit", "memory_limit", 6)
        
        # Button frame
        button_frame = ttk.Frame(self.dialog, padding="10")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(button_frame, text="Save", command=self._on_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)
    
    def _create_field(self, label_text: str, config_key: str, row: int):
        """Create a label and entry field for a config option."""
        ttk.Label(self.settings_frame, text=label_text, font=('Segoe UI', 9, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(5, 0))
        
        if config_key in ['tavily_max_results', 'memory_limit']:
            # Integer fields
            entry = ttk.Entry(self.settings_frame, width=50)
            entry.insert(0, str(self.config.get(config_key, "")))
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
            setattr(self, f"_{config_key}_entry", entry)
        else:
            # String fields (password fields for API keys)
            if 'api_key' in config_key:
                entry = ttk.Entry(self.settings_frame, width=50, show='*')
            else:
                entry = ttk.Entry(self.settings_frame, width=50)
            entry.insert(0, self.config.get(config_key, ""))
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
            setattr(self, f"_{config_key}_entry", entry)
        
        self.settings_frame.grid_columnconfigure(1, weight=1)
    
    def _on_save(self):
        """Handle save button click."""
        logger.debug("SettingsDialog: Save button clicked")
        # Parse models from text area
        models_text = self._models_text.get('1.0', tk.END).strip()
        models = [m.strip() for m in models_text.split('\n') if m.strip()]
        if not models:
            models = ["qwen/qwen3.6-flash"]
        
        active_model = self._active_model_var.get().strip()
        if not active_model:
            active_model = models[0] if models else "qwen/qwen3.6-flash"
        
        self.result = {
            "routerapi_base_url": self._routerapi_base_url_entry.get().strip(),
            "routerapi_api_key": self._routerapi_api_key_entry.get().strip(),
            "models": models,
            "active_model": active_model,
            "tavily_api_key": self._tavily_api_key_entry.get().strip(),
            "tavily_max_results": int(self._tavily_max_results_entry.get().strip()),
            "memory_limit": int(self._memory_limit_entry.get().strip())
        }
        logger.debug(f"SettingsDialog: New config keys: {list(self.result.keys())}")
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        logger.debug("SettingsDialog: Cancel button clicked")
        self.result = None
        self.dialog.destroy()


class AIChatGUI:
    """Main GUI application for the AI Agent."""
    
    def __init__(self, root: tk.Tk):
        logger.debug("AIChatGUI: Initializing GUI application")
        self.root = root
        self.root.title("AI Agent Chat")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Initialize config manager
        self.config_manager = ConfigManager()
        
        # Load configuration
        self.config = self.config_manager.load_config()
        logger.debug(f"AIChatGUI: Loaded config with keys: {list(self.config.keys())}")
        
        # Ensure models list and active_model exist in config
        if "models" not in self.config:
            self.config["models"] = ["qwen/qwen3.6-flash"]
        if "active_model" not in self.config:
            self.config["active_model"] = self.config["models"][0] if self.config["models"] else "qwen/qwen3.6-flash"
        
        # Initialize agent with loaded config, falling back to environment variables
        try:
            logger.debug("AIChatGUI: Initializing AIAgent")
            self.agent = AIAgent(
                routerapi_base_url=self.config.get("routerapi_base_url") or os.getenv("ROUTERAPI_BASE_URL"),
                routerapi_api_key=self.config.get("routerapi_api_key") or os.getenv("ROUTERAPI_API_KEY"),
                tavily_api_key=self.config.get("tavily_api_key") or os.getenv("TAVILY_API_KEY"),
                model_name=self.config.get("active_model") or os.getenv("MODEL_NAME", "qwen/qwen3.6-flash"),
                tavily_max_results=self.config.get("tavily_max_results"),
                memory_limit=self.config.get("memory_limit")
            )
            logger.debug("AIChatGUI: AIAgent initialized successfully")
        except Exception as e:
            logger.error(f"AIChatGUI: Failed to initialize AIAgent: {e}", exc_info=True)
            self.agent = None
            self.agent_error = str(e)
        
        # Chat history
        self.chat_history: list[ChatMessage] = []
        
        # Selected files for upload
        self.selected_files: List[str] = []
        
        # Model switcher variable
        self._model_switcher_var = tk.StringVar()
        self._model_switcher_var.trace('w', self._on_model_changed)
        
        # Setup UI
        self._setup_ui()
        
        # Hide file list widgets on startup (no files attached yet)
        self._update_file_list_display()
        
        # Start with a welcome message
        self._add_system_message("Welcome to AI Agent Chat!")
        if self.agent is None:
            self._add_error_message(f"Agent initialization error: {self.agent_error}")
        logger.debug("AIChatGUI: GUI initialization complete")
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Create menu bar
        self._setup_menu_bar()
        
        # Create chat display area
        self._setup_chat_display(main_container)
        
        # Create input area (includes model switcher and buttons)
        self._setup_input_area(main_container)
        
        # Create status bar
        self._setup_status_bar()
    
    def _setup_menu_bar(self):
        """Setup the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear Chat", command=self._clear_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure", command=self._open_settings)
        settings_menu.add_separator()
        settings_menu.add_command(label="Reset Agent", command=self._reset_agent)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _setup_chat_display(self, parent: ttk.Frame):
        """Setup the chat display area."""
        # Create a frame for the chat display
        chat_frame = ttk.Frame(parent)
        chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(chat_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        # Create scrolled text widget
        self.chat_display = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            state='disabled',
            font=('Segoe UI', 10),
            padx=10,
            pady=10,
            relief='sunken',
            borderwidth=1
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure message tags
        self._configure_message_tags()
        
        # Bind keyboard shortcuts
        self.chat_display.bind('<Control-a>', lambda e: self._select_all())
        self.chat_display.bind('<Control-c>', lambda e: self._copy_selection())
        self.chat_display.bind('<Button-3>', lambda e: self._show_context_menu(e))
        
        # Setup context menu
        self._setup_context_menu()
    
    def _configure_message_tags(self):
        """Configure tags for different message types."""
        # User message
        self.chat_display.tag_config(
            'user_message',
            background='#e3f2fd',
            foreground='#000000',
            font=('Segoe UI', 10, 'bold'),
            lmargin1=20,
            lmargin2=20,
            rmargin=20,
            spacing1=5,
            spacing3=5
        )
        
        # Agent message
        self.chat_display.tag_config(
            'agent_message',
            background='#f5f5f5',
            foreground='#000000',
            font=('Segoe UI', 10),
            lmargin1=20,
            lmargin2=20,
            rmargin=20,
            spacing1=5,
            spacing3=5
        )
        
        # System message
        self.chat_display.tag_config(
            'system_message',
            background='#fff3e0',
            foreground='#e65100',
            font=('Segoe UI', 9, 'italic'),
            lmargin1=20,
            lmargin2=20,
            rmargin=20,
            spacing1=5,
            spacing3=5
        )
        
        # Error message
        self.chat_display.tag_config(
            'error_message',
            background='#ffebee',
            foreground='#c62828',
            font=('Segoe UI', 9),
            lmargin1=20,
            lmargin2=20,
            rmargin=20,
            spacing1=5,
            spacing3=5
        )
        
        # Markdown content (for agent responses)
        self.chat_display.tag_config(
            'markdown_content',
            font=('Segoe UI', 10)
        )
        
        # Timestamp
        self.chat_display.tag_config(
            'timestamp',
            font=('Segoe UI', 8),
            foreground='#666666'
        )
    
    def _on_model_changed(self, *args):
        """Handle model change event."""
        new_model = self._model_switcher_var.get().strip()
        if not new_model:
            return
        
        logger.debug(f"AIChatGUI: Model changed to {new_model}")
        
        # Update config
        self.config["active_model"] = new_model
        
        # Reinitialize agent with new model
        try:
            logger.debug("AIChatGUI: Reinitializing agent with new model")
            self.agent = AIAgent(
                routerapi_base_url=self.config.get("routerapi_base_url") or os.getenv("ROUTERAPI_BASE_URL"),
                routerapi_api_key=self.config.get("routerapi_api_key") or os.getenv("ROUTERAPI_API_KEY"),
                tavily_api_key=self.config.get("tavily_api_key") or os.getenv("TAVILY_API_KEY"),
                model_name=new_model,
                tavily_max_results=self.config.get("tavily_max_results"),
                memory_limit=self.config.get("memory_limit")
            )
            self._add_system_message(f"Switched to model: {new_model}")
            logger.debug("AIChatGUI: Agent reinitialized with new model")
        except Exception as e:
            logger.error(f"AIChatGUI: Failed to reinitialize agent with new model: {e}", exc_info=True)
            self._add_error_message(f"Failed to switch model: {str(e)}")
    
    def _setup_input_area(self, parent: ttk.Frame):
        """Setup the input area."""
        # Create input frame
        input_frame = ttk.Frame(parent)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Create input text widget
        self.input_text = tk.Text(
            input_frame,
            height=4,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            padx=10,
            pady=10
        )
        self.input_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.input_text.bind('<Return>', lambda e: self._on_enter_pressed(e))
        
        # Create file list frame (below input, above model switcher)
        file_list_frame = ttk.Frame(input_frame)
        file_list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        file_list_frame.grid_columnconfigure(0, weight=1)
        
        # File list label
        self.file_list_label = ttk.Label(file_list_frame, text="Attached Files:", font=('Segoe UI', 9))
        self.file_list_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        # File list display (read-only)
        self.file_list_display = tk.Text(
            file_list_frame,
            height=3,
            wrap=tk.WORD,
            font=('Segoe UI', 9),
            state='disabled',
            background='#f5f5f5',
            relief='sunken',
            borderwidth=1
        )
        self.file_list_display.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # File list scrollbar
        self.file_list_scrollbar = ttk.Scrollbar(
            file_list_frame,
            orient="vertical",
            command=self.file_list_display.yview
        )
        self.file_list_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.file_list_display.configure(yscrollcommand=self.file_list_scrollbar.set)
        
        # File buttons frame
        file_button_frame = ttk.Frame(file_list_frame)
        file_button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        file_button_frame.grid_columnconfigure(0, weight=1)
        
        # Add file button
        self.add_file_button = ttk.Button(
            file_button_frame,
            text="+ Add File",
            command=self._add_file
        )
        self.add_file_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear files button
        self.clear_files_button = ttk.Button(
            file_button_frame,
            text="Clear Files",
            command=self._clear_files
        )
        self.clear_files_button.pack(side=tk.LEFT)
        
        # Create model switcher frame (below file area)
        model_frame = ttk.Frame(input_frame)
        model_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        model_frame.grid_columnconfigure(1, weight=1)
        
        # Model label
        ttk.Label(model_frame, text="Model:", font=('Segoe UI', 9)).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        # Model combobox
        models = self.config.get("models", ["qwen/qwen3.6-flash"])
        self._model_switcher_var.set(self.config.get("active_model", models[0] if models else ""))
        self._model_switcher_combo = ttk.Combobox(
            model_frame,
            textvariable=self._model_switcher_var,
            state="readonly"
        )
        self._model_switcher_combo['values'] = models
        self._model_switcher_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self._model_switcher_combo.bind('<<ComboboxSelected>>', lambda e: self._on_model_changed())
        
        # Create button frame (below model switcher)
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Send button
        self.send_button = ttk.Button(
            button_frame,
            text="Send",
            command=self._send_message
        )
        self.send_button.grid(row=0, column=0, sticky=tk.E)
        
        # Clear input button
        clear_button = ttk.Button(
            button_frame,
            text="Clear",
            command=self._clear_input
        )
        clear_button.grid(row=0, column=1, sticky=tk.E, padx=(5, 0))
        
        # Bind focus
        self.input_text.focus_set()
    
    def _setup_status_bar(self):
        """Setup the status bar."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor=tk.W,
            padding=(5, 2)
        )
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def _on_enter_pressed(self, event):
        """Handle Enter key press."""
        if event.state & 0x1:  # Shift key is pressed
            return False  # Allow newline
        
        self._send_message()
        return 'break'
    
    def _send_message(self):
        """Send a message to the agent."""
        user_input = self.input_text.get(1.0, tk.END).strip()
        logger.debug(f"AIChatGUI: User sent message (length: {len(user_input)} chars)")
        
        if not user_input and not self.selected_files:
            logger.debug("AIChatGUI: Empty message and no files, ignoring")
            return
        
        # Copy file paths BEFORE clearing input (since _clear_input clears selected_files)
        file_paths_to_process = self.selected_files.copy()
        
        # Display user message
        self._add_user_message(user_input)
        
        # Clear input (this will clear selected_files)
        self._clear_input()
        
        # Process in background thread with copied file paths
        logger.debug("AIChatGUI: Starting background thread to process query")
        threading.Thread(
            target=self._process_user_query,
            args=(user_input,),
            kwargs={'file_paths': file_paths_to_process},
            daemon=True
        ).start()
    
    def _process_user_query(self, query: str, file_paths: Optional[List[str]] = None):
        """Process user query in background thread."""
        logger.debug(f"AIChatGUI: Processing user query: {query[:50]}...")
        if file_paths:
            logger.debug(f"AIChatGUI: Attached files: {file_paths}")
        self.status_var.set("Processing...")
        
        try:
            if self.agent is None:
                logger.error("AIChatGUI: Agent is not initialized")
                raise Exception("Agent not initialized")
            
            # Get response from agent
            logger.debug("AIChatGUI: Calling agent.run()")
            response = self.agent.run(query, file_paths=file_paths)
            logger.debug(f"AIChatGUI: Received response from agent (length: {len(response)} chars)")
            
            # Display agent response
            self._add_agent_message(response)
            
        except Exception as e:
            logger.error(f"AIChatGUI: Error processing query: {e}", exc_info=True)
            self._add_error_message(f"Error: {str(e)}")
        
        finally:
            self.status_var.set("Ready")
            logger.debug("AIChatGUI: Query processing complete")
    
    def _add_user_message(self, content: str):
        """Add a user message to the chat."""
        timestamp = self._get_timestamp()
        
        self.chat_display.config(state='normal')
        
        # Add timestamp
        self.chat_display.insert(tk.END, f"[{timestamp}] You:\n", 'timestamp')
        
        # Add message content
        self.chat_display.insert(tk.END, content + '\n', 'user_message')
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
        # Store in history
        self.chat_history.append(ChatMessage('user', content))
    
    def _add_agent_message(self, content: str):
        """Add an agent message to the chat with markdown formatting."""
        timestamp = self._get_timestamp()

        self.chat_display.config(state='normal')

        # Add timestamp
        self.chat_display.insert(tk.END, f"[{timestamp}] Agent:\n", 'timestamp')

        # Add message content with markdown formatting
        markdown_converter = MarkdownToTkinter(self.chat_display)
        markdown_converter.insert_markdown(content)

        # Separator blank line after the message
        self.chat_display.insert(tk.END, '\n', 'normal')

        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

        # Store in history
        self.chat_history.append(ChatMessage('agent', content))
    
    def _add_system_message(self, content: str):
        """Add a system message to the chat."""
        timestamp = self._get_timestamp()
        
        self.chat_display.config(state='normal')
        self.chat_display.insert(
            tk.END,
            f"[{timestamp}] System: {content}\n",
            'system_message'
        )
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
        self.chat_history.append(ChatMessage('system', content))
    
    def _add_error_message(self, content: str):
        """Add an error message to the chat."""
        timestamp = self._get_timestamp()
        
        self.chat_display.config(state='normal')
        self.chat_display.insert(
            tk.END,
            f"[{timestamp}] Error: {content}\n",
            'error_message'
        )
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
        self.chat_history.append(ChatMessage('error', content))
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")
    
    def _clear_chat(self):
        """Clear the chat display."""
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state='disabled')
        self.chat_history.clear()
        self._add_system_message("Chat cleared")
    
    def _clear_input(self):
        """Clear the input text and attached files."""
        self.input_text.delete(1.0, tk.END)
        self.selected_files.clear()
        self._update_file_list_display()
    
    def _add_file(self):
        """Add a file to the attachment list."""
        logger.debug("AIChatGUI: Opening file dialog")
        file_paths = filedialog.askopenfilenames(
            title="Select files to attach",
            filetypes=[
                ("All supported files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.tiff;*.pdf;*.txt;*.md;*.json;*.csv"),
                ("Images", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.tiff"),
                ("Documents", "*.pdf;*.txt;*.md"),
                ("Data files", "*.json;*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            logger.debug(f"AIChatGUI: Selected {len(file_paths)} files")
            for file_path in file_paths:
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)
            self._update_file_list_display()
    
    def _clear_files(self):
        """Clear all attached files."""
        logger.debug("AIChatGUI: Clearing attached files")
        self.selected_files.clear()
        self._update_file_list_display()
    
    def _update_file_list_display(self):
        """Update the file list display widget."""
        logger.debug(f"AIChatGUI: Updating file list display with {len(self.selected_files)} files")
        
        if not self.selected_files:
            # Hide file list widgets when no files attached
            self.file_list_label.grid_remove()
            self.file_list_display.grid_remove()
            self.file_list_scrollbar.grid_remove()
        else:
            # Show file list widgets when files are attached
            self.file_list_label.grid()
            self.file_list_display.grid()
            self.file_list_scrollbar.grid()
            
            self.file_list_display.config(state='normal')
            self.file_list_display.delete(1.0, tk.END)
            for i, file_path in enumerate(self.selected_files, 1):
                file_name = os.path.basename(file_path)
                self.file_list_display.insert(tk.END, f"{i}. {file_name}\n")
            self.file_list_display.config(state='disabled')
            self.file_list_display.see(tk.END)
    
    def _open_settings(self):
        """Open settings dialog to edit configuration."""
        logger.debug("AIChatGUI: Opening settings dialog")
        dialog = SettingsDialog(self.root, self.config)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            logger.debug("AIChatGUI: Settings saved")
            # Update config with new values
            self.config = dialog.result
            
            # Save config to file
            if self.config_manager.save_config(self.config):
                self._add_system_message("Settings saved successfully")
                logger.debug("AIChatGUI: Config saved to file")
                
                # Update model switcher combobox values
                models = self.config.get("models", ["qwen/qwen3.6-flash"])
                self._model_switcher_combo['values'] = models
                
                # Update active model if needed
                active_model = self.config.get("active_model", models[0] if models else "")
                if active_model != self._model_switcher_var.get():
                    self._model_switcher_var.set(active_model)
                
                # Update agent with new settings, falling back to environment variables
                try:
                    logger.debug("AIChatGUI: Reinitializing agent with new settings")
                    self.agent = AIAgent(
                        routerapi_base_url=self.config.get("routerapi_base_url") or os.getenv("ROUTERAPI_BASE_URL"),
                        routerapi_api_key=self.config.get("routerapi_api_key") or os.getenv("ROUTERAPI_API_KEY"),
                        tavily_api_key=self.config.get("tavily_api_key") or os.getenv("TAVILY_API_KEY"),
                        model_name=active_model,
                        tavily_max_results=self.config.get("tavily_max_results"),
                        memory_limit=self.config.get("memory_limit")
                    )
                    self._add_system_message("Agent reinitialized with new settings")
                    logger.debug("AIChatGUI: Agent reinitialized successfully")
                except Exception as e:
                    logger.error(f"AIChatGUI: Failed to reinitialize agent: {e}", exc_info=True)
                    self._add_error_message(f"Failed to reinitialize agent: {str(e)}")
            else:
                logger.error("AIChatGUI: Failed to save config to file")
                self._add_error_message("Failed to save settings")
        else:
            logger.debug("AIChatGUI: Settings dialog cancelled")
    
    def _reset_agent(self):
        """Reset the agent."""
        logger.debug("AIChatGUI: Resetting agent memory")
        try:
            self.agent.clear_memory()
            self._add_system_message("Agent memory cleared")
            self.status_var.set("Agent reset")
            logger.debug("AIChatGUI: Agent memory cleared successfully")
        except Exception as e:
            logger.error(f"AIChatGUI: Error resetting agent: {e}", exc_info=True)
            self._add_error_message(f"Error resetting agent: {str(e)}")
    
    def _show_about(self):
        """Show about dialog."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About AI Agent Chat")
        about_window.geometry("400x200")
        about_window.transient(self.root)
        about_window.grab_set()
        
        # Center window
        about_window.update_idletasks()
        x = (about_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (about_window.winfo_screenheight() // 2) - (200 // 2)
        about_window.geometry(f"400x200+{x}+{y}")
        
        # About content
        ttk.Label(
            about_window,
            text="AI Agent Chat",
            font=('Segoe UI', 16, 'bold')
        ).pack(pady=10)
        
        ttk.Label(
            about_window,
            text="A GUI application for interacting with AI Agent",
            font=('Segoe UI', 10)
        ).pack(pady=5)
        
        ttk.Label(
            about_window,
            text="Features:\n"
                 "- Markdown support for formatted responses\n"
                 "- Real-time chat interface\n"
                 "- Conversation memory\n"
                 "- Tool integration (search, math)",
            font=('Segoe UI', 9),
            justify=tk.LEFT
        ).pack(pady=10)
        
        ttk.Button(
            about_window,
            text="Close",
            command=about_window.destroy
        ).pack(pady=10)
    
    def _setup_context_menu(self):
        """Setup right-click context menu for the chat display."""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self._copy_selection)
        self.context_menu.add_command(label="Select All", command=self._select_all)
    
    def _show_context_menu(self, event):
        """Show context menu on right-click."""
        try:
            self.chat_display.config(state='normal')
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.chat_display.config(state='disabled')
    
    def _select_all(self):
        """Select all text in chat display."""
        self.chat_display.config(state='normal')
        self.chat_display.tag_add(tk.SEL, '1.0', tk.END)
        self.chat_display.mark_set(tk.INSERT, '1.0')
        self.chat_display.see(tk.INSERT)
        self.chat_display.config(state='disabled')
        return 'break'
    
    def _copy_selection(self):
        """Copy selected text to clipboard."""
        try:
            self.chat_display.config(state='normal')
            selection = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selection)
            self.chat_display.config(state='disabled')
        except tk.TclError:
            pass  # No selection


def main():
    """Main function to run the GUI application."""
    logger.debug("GUI: Starting main function")
    root = tk.Tk()
    logger.debug("GUI: Created root window")
    app = AIChatGUI(root)
    logger.debug("GUI: Created AIChatGUI instance")
    root.mainloop()
    logger.debug("GUI: Main loop exited")


if __name__ == "__main__":
    main()
