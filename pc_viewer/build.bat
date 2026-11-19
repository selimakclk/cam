@echo off
zig cc -target x86_64-windows-gnu -Os -s -Wall -Wextra -Wno-unused-parameter "-Wl,--subsystem,windows" "%~dp0cam_viewer.c" -o "%~dp0..\CAM_Izleyici.exe" -luser32 -lgdi32 -lcomctl32 -lws2_32 -liphlpapi -lwinmm -lole32 -lwindowscodecs -ldwmapi
if %errorlevel%==0 (
    echo Derleme basarili: CAM_Izleyici.exe
) else (
    echo Derleme hatasi!
)
