@echo off

set OBS=%~dp0obs64.exe
echo %OBS%

mkdir cl_cache
set CL_CACHE=%~dp0cl_cache
echo %CL_CACHE%

set REG=HKEY_LOCAL_MACHINE\SOFTWARE\Intel\IGFX\OCL\cl_cache_dir
echo %REG%

reg add %REG% /v %OBS% /t REG_SZ /d %CL_CACHE%

pause


