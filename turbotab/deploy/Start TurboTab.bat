@echo off
title TurboTab
rem TurboTab: double-click to start it on Windows.
rem
rem The first time, Windows may warn about a file downloaded from the internet: choose
rem More info, then Run anyway. The first start sets TurboTab up (a few minutes, once); after
rem that it takes seconds. Keep the window open while you work; closing it stops TurboTab.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0turbotab.ps1" %*
if errorlevel 1 (
    echo.
    echo TurboTab stopped with an error; the message above says why.
    if not defined CI pause
    exit /b 1
)
