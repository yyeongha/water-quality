# pyqt

for windows ( linux, macOS not tested )
```
pip install -r requirements.txt
```

### start project
```
fbs run
```

### start build
```
fbs freeze
```

bug
```
FileNotFoundError: Could not find msvcr110.dll on your PATH. Please install the Visual C++ Redistributable for Visual Studio 2012 from:
```

bugfix
```
https://www.microsoft.com/en-us/download/details.aspx?id=30679
```

### start deploy
```
fbs installer
```
bug
```
FileNotFoundError: fbs could not find executable 'makensis'. Please install NSIS and add its installation directory to your PATH environment variable.
```
bugfix
```
https://sourceforge.net/projects/nsis/
```
install nsis and set env path

### reference

fbs tutorial
https://github.com/mherrmann/fbs-tutorial