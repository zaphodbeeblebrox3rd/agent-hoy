#!/usr/bin/env python3
"""
Test script to verify the resizable layout works correctly
"""

import tkinter as tk
from tkinter import ttk, scrolledtext

def test_paned_window():
    """Test the three-pane layout with vertical and horizontal splits"""
    root = tk.Tk()
    root.title("Three-Pane Layout Test")
    root.geometry("1000x700")
    
    # Create main frame
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Control frame (simulating the top control bar)
    control_frame = ttk.Frame(main_frame)
    control_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Simulate the control buttons and status
    start_button = ttk.Button(control_frame, text="Start Listening")
    start_button.pack(side=tk.LEFT, padx=(0, 10))
    
    stop_button = ttk.Button(control_frame, text="Stop Listening")
    stop_button.pack(side=tk.LEFT, padx=(0, 10))
    
    clear_button = ttk.Button(control_frame, text="Clear Text")
    clear_button.pack(side=tk.LEFT, padx=(0, 10))
    
    status_label = ttk.Label(control_frame, text="Status: Ready")
    status_label.pack(side=tk.LEFT, padx=(20, 0))
    
    audio_status_label = ttk.Label(control_frame, text="Audio Status: Default Microphone")
    audio_status_label.pack(side=tk.LEFT, padx=(20, 0))
    
    ai_checkbox = ttk.Checkbutton(control_frame, text="Enable AI Analysis")
    ai_checkbox.pack(side=tk.LEFT, padx=(20, 0))
    
    openai_status_label = ttk.Label(control_frame, text="OpenAI: Template Mode")
    openai_status_label.pack(side=tk.LEFT, padx=(20, 0))
    
    cost_label = ttk.Label(control_frame, text="Session Cost: $0.00")
    cost_label.pack(side=tk.LEFT, padx=(20, 0))
    
    cost_reset_button = ttk.Button(control_frame, text="Reset Cost")
    cost_reset_button.pack(side=tk.LEFT, padx=(0, 10))
    
    # Create vertical PanedWindow (transcription vs bottom)
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
    
    # Bottom pane - Split into topic explanation and AI output
    bottom_frame = ttk.Frame(paned_window)
    paned_window.add(bottom_frame, weight=1)
    
    # Create horizontal PanedWindow for bottom split
    bottom_paned = ttk.PanedWindow(bottom_frame, orient=tk.HORIZONTAL)
    bottom_paned.pack(fill=tk.BOTH, expand=True)
    
    # Left side - Topic explanation
    topic_frame = ttk.LabelFrame(bottom_paned, text="Topic Explanation & Troubleshooting", padding="5")
    bottom_paned.add(topic_frame, weight=1)
    
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
    topic_text.insert(tk.END, "This is the topic explanation area (LEFT).\n\n" +
                     "This contains hardcoded topic explanations with commands.\n\n" +
                     "You can drag the divider to resize between topic and AI panes.\n\n" +
                     "This area shows structured information about clicked topics.")
    topic_text.config(state=tk.DISABLED)
    
    # Right side - AI output
    ai_frame = ttk.LabelFrame(bottom_paned, text="AI-Driven Analysis & Suggestions", padding="5")
    bottom_paned.add(ai_frame, weight=1)
    
    ai_text = scrolledtext.ScrolledText(
        ai_frame,
        wrap=tk.WORD,
        height=15,
        font=("Arial", 10),
        state=tk.DISABLED
    )
    ai_text.pack(fill=tk.BOTH, expand=True)
    
    # Enable AI text for testing
    ai_text.config(state=tk.NORMAL)
    ai_text.insert(tk.END, "This is the AI-driven analysis area (RIGHT).\n\n" +
                  "This contains AI-enhanced insights and suggestions.\n\n" +
                  "You can drag the divider to resize between topic and AI panes.\n\n" +
                  "This area shows advanced analysis and recommendations.")
    ai_text.config(state=tk.DISABLED)
    
    # Set equal weights for all panes
    paned_window.pane(0, weight=1)
    paned_window.pane(1, weight=1)
    bottom_paned.pane(0, weight=1)
    bottom_paned.pane(1, weight=1)
    
    # Add some test content to make scrolling visible
    for i in range(20):
        transcription_text.insert(tk.END, f"Line {i+1}: This is test content to demonstrate scrolling functionality.\n")
    
    print("Three-pane layout test window created. You should see:")
    print("1. Control bar with buttons and status labels")
    print("2. Audio Status: Default Microphone")
    print("3. Enable AI Analysis checkbox")
    print("4. OpenAI: Template Mode status")
    print("5. Session Cost: $0.00 indicator")
    print("6. Reset Cost button")
    print("7. Top pane: Transcription area")
    print("8. Bottom left: Topic explanation area")
    print("9. Bottom right: AI-driven analysis area")
    print("10. Vertical divider between top and bottom")
    print("11. Horizontal divider between left and right bottom panes")
    print("12. All panes are resizable by dragging dividers")
    print("13. Scrollbars in all areas when needed")
    
    root.mainloop()

if __name__ == "__main__":
    test_paned_window()
