; Inno Setup script for the LiteNotebook Windows installer.
; Built by tools/build.py on Windows:  ISCC /DAppVersion=... /DAppFolder=... litenotebook.iss
; Per-user install (no administrator rights). Uninstall removes the program
; files only: notebooks, models, settings, logs and audio in
; %LOCALAPPDATA%\LiteNotebook are never deleted.

#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef AppFolder
  #error AppFolder (the PyInstaller output folder) is required
#endif

[Setup]
AppId={{6C1E9E5A-2C3B-4F35-9A7E-0B7D2C9F4A11}
AppName=LiteNotebook
AppVersion={#AppVersion}
AppVerName=LiteNotebook {#AppVersion}
AppPublisher=LiteNotebook
DefaultDirName={localappdata}\Programs\LiteNotebook
DefaultGroupName=LiteNotebook
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=LiteNotebook-{#AppVersion}-Setup
#ifdef IconFile
SetupIconFile={#IconFile}
#endif
UninstallDisplayIcon={app}\LiteNotebook.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Types]
Name: "full"; Description: "LiteNotebook with semantic search runtime"
Name: "core"; Description: "LiteNotebook only (install semantic search later)"

[Components]
Name: "app"; Description: "LiteNotebook"; Types: full core; Flags: fixed
#ifdef RuntimeFolder
Name: "runtime"; Description: "Semantic search runtime (PyTorch CPU, needed to index PDFs; large)"; Types: full
#endif

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#AppFolder}\*"; DestDir: "{app}"; Components: app; Flags: recursesubdirs createallsubdirs ignoreversion
#ifdef RuntimeFolder
Source: "{#RuntimeFolder}\*"; DestDir: "{app}\embedding-runtime"; Components: runtime; Flags: recursesubdirs createallsubdirs ignoreversion
#endif

[Icons]
Name: "{group}\LiteNotebook"; Filename: "{app}\LiteNotebook.exe"
Name: "{group}\LiteNotebook logs folder"; Filename: "{localappdata}\LiteNotebook\Logs"
Name: "{autodesktop}\LiteNotebook"; Filename: "{app}\LiteNotebook.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\LiteNotebook.exe"; Description: "Start LiteNotebook"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; program folder only (files the installer created); user data is untouched
Type: filesandordirs; Name: "{app}\embedding-runtime"
