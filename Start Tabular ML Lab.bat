@echo off
title Tabular ML Lab
rem Tabular ML Lab — Windows starter.
rem First time: Windows SmartScreen may ask once ("More info" -> "Run anyway")
rem because this file was downloaded from the internet. Setup runs once
rem (~5 minutes), then the app opens in your browser and a "Tabular ML Lab"
rem shortcut appears on your Desktop and Start Menu for future launches.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\windows_setup.ps1"
if errorlevel 1 (
    echo.
    echo Something went wrong. Please screenshot the message above when asking for help.
    pause
)
