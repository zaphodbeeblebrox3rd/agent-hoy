#!/usr/bin/env python3
"""
Test script to verify the resizable layout works correctly
"""

import tkinter as tk
from tkinter import ttk, scrolledtext

def test_paned_window():
    """Test the PanedWindow layout"""
    root = tk.Tk()
    root.title("Layout Test")
    root.geometry("800x600")
    
    # Create main frame
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create PanedWindow
    paned_window = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
    paned_window.pack(fill=tk.BOTH, expand=True)
    
    # Top pane - Transcription
    transcription_frame = ttk.LabelFrame(paned_window, text="Live Transcription", padding="5")
    paned_window.add(transcription_frame, weight=1)
    
    transcription_text = scrolledtext.ScrolledText(
        transcription_frame, 
        wrap=tk.WORD, 
        height=15,
        font=("Arial", 12)
    )
    transcription_text.pack(fill=tk.BOTH, expand=True)
    transcription_text.insert(tk.END, "This is the transcription area. You can type here to test scrolling.\n\n" + 
                            "This area should be resizable by dragging the divider between the two panes.\n\n" +
                            "The scrollbar should appear when content exceeds the visible area.")
    
    # Bottom pane - Topic explanation
    topic_frame = ttk.LabelFrame(paned_window, text="Topic Explanation & Troubleshooting", padding="5")
    paned_window.add(topic_frame, weight=1)
    
    topic_text = scrolledtext.ScrolledText(
        topic_frame,
        wrap=tk.WORD,
        height=15,
        font=("Arial", 10),
        state=tk.DISABLED
    )
    topic_text.pack(fill=tk.BOTH, expand=True)
    
    # Enable topic text for testing
    topic_text.config(state=tk.NORMAL)
    topic_text.insert(tk.END, "This is the topic explanation area. It should be the same size as the transcription area.\n\n" +
                     "You can drag the divider between the two panes to resize them.\n\n" +
                     "Both areas have scrollbars that appear when needed.\n\n" +
                     "This area is initially disabled but can be enabled for editing.")
    topic_text.config(state=tk.DISABLED)
    
    # Set equal weights
    paned_window.pane(0, weight=1)
    paned_window.pane(1, weight=1)
    
    # Add some test content to make scrolling visible
    for i in range(20):
        transcription_text.insert(tk.END, f"Line {i+1}: This is test content to demonstrate scrolling functionality.\n")
    
    print("Layout test window created. You should see:")
    print("1. Two resizable panes (transcription and topic explanation)")
    print("2. Scrollbars in both areas")
    print("3. Ability to drag the divider to resize panes")
    print("4. Equal initial sizes for both panes")
    
    root.mainloop()

if __name__ == "__main__":
    test_paned_window()
