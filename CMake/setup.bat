rmdir build /S /Q
mkdir build
cd build
cmake ..
cmake --build . --config Release

cd Release
gvs.exe

pause