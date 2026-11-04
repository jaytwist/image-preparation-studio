@echo off
rem Double-click launcher: runs run.ps1 without changing the system execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
