@echo off

call rmdir build /s /q
call cmake -S . -B build %*
