@echo off
rem Start Edge with the debugging port so agent-hoy can read the page
rem instead of relying on screenshots. Edge must be fully closed first:
rem Windows ignores the flag when an Edge process is already running,
rem including the hidden background instance Edge sometimes keeps alive.
tasklist /FI "IMAGENAME eq msedge.exe" | find /I "msedge.exe" >nul
if not errorlevel 1 (
    echo Edge is already running. The debugging port only works when Edge
    echo starts with it. Close every Edge window ^(and check the system tray
    echo and Task Manager for a background copy^), then run this again.
    pause
    exit /b 1
)
start "" "msedge.exe" --remote-debugging-port=9222
echo Edge started with debugging port 9222. agent-hoy can attach now.
