@echo off
echo Activating Emscripten Compiler...
call C:\Users\ABI-AI\Desktop\emsdk\emsdk_env.bat
echo Compiling C++ Core Engine to WebAssembly...
if not exist public mkdir public
call emcc -O3 cpp/engine.cpp -o public/engine.js -std=c++17 --bind -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORT_ES6=1 -s MODULARIZE=1
echo Compilation finished.
