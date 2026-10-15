; Inno Setup script for the LiteNotebook Windows installer (Inno Setup 6.4+).
; Built by tools/build.py on Windows:
;   ISCC /DAppVersion=... /DAppFolder=... [/DRuntimeFolder=...] litenotebook.iss
;
; Wizard: Welcome > Information (licence and notices) > Install location >
; Components (LiteNotebook; optional semantic-search runtime) > Shortcuts >
; Ready > Progress > Finish (Launch LiteNotebook).
;
; Per-user install, no administrator rights. No file associations. Nothing is
; downloaded. Uninstall removes the program files and shortcuts only:
; notebooks, PDFs, notes, chats, backups, models and AI components in
; %LOCALAPPDATA%\LiteNotebook are never deleted.
; Keep this file ASCII-only (Inno Setup reads it with the system code page).

#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef AppFolder
  #error AppFolder (the PyInstaller output folder) is required
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
#define AppExe "LiteNotebook.exe"

[Setup]
; Never change AppId: upgrades find the earlier installation through it.
AppId={{6C1E9E5A-2C3B-4F35-9A7E-0B7D2C9F4A11}
AppName=LiteNotebook
AppVersion={#AppVersion}
AppVerName=LiteNotebook {#AppVersion}
AppPublisher=LiteNotebook
VersionInfoVersion={#AppVersion}
VersionInfoProductName=LiteNotebook
VersionInfoProductVersion={#AppVersion}
VersionInfoDescription=LiteNotebook Setup
UninstallDisplayName=LiteNotebook
UninstallDisplayIcon={app}\{#AppExe}
DefaultDirName={localappdata}\Programs\LiteNotebook
UsePreviousAppDir=yes
DefaultGroupName=LiteNotebook
DisableProgramGroupPage=yes
DisableWelcomePage=no
DisableDirPage=no
DisableReadyPage=no
InfoBeforeFile={#SourcePath}\before-install.txt
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=LiteNotebook-Setup-{#AppVersion}-x64
#ifdef IconFile
SetupIconFile={#IconFile}
#endif
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
ChangesAssociations=no
SetupLogging=yes

[Types]
Name: "custom"; Description: "Custom"; Flags: iscustom

[Components]
Name: "app"; Description: "LiteNotebook (required)"; Types: custom; Flags: fixed
#ifdef RuntimeFolder
; In no type: unticked on a first install; an upgrade keeps the earlier choice.
Name: "runtime"; Description: "Install local semantic-search runtime: enables local embeddings and semantic search (needed to add new PDFs). Approximately 1 GB installed."
#endif

[Tasks]
Name: "startmenuicon"; Description: "Create Start Menu shortcut"; GroupDescription: "Shortcuts:"
Name: "desktopicon"; Description: "Create Desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[InstallDelete]
; Program files of an earlier version only, so an upgrade never mixes old and
; new libraries. The data folder (%LOCALAPPDATA%\LiteNotebook) is elsewhere.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\embedding-runtime"

[Files]
Source: "{#AppFolder}\*"; DestDir: "{app}"; Components: app; Flags: recursesubdirs createallsubdirs ignoreversion
#ifdef RuntimeFolder
Source: "{#RuntimeFolder}\*"; DestDir: "{app}\embedding-runtime"; Components: runtime; Flags: recursesubdirs createallsubdirs ignoreversion
#endif

[Icons]
Name: "{group}\LiteNotebook"; Filename: "{app}\{#AppExe}"; Tasks: startmenuicon
Name: "{group}\LiteNotebook logs folder"; Filename: "{localappdata}\LiteNotebook\Logs"; Tasks: startmenuicon
Name: "{autodesktop}\LiteNotebook"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch LiteNotebook"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; program folder only; user data is untouched
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\embedding-runtime"

[Code]
function DataFolder(): String;
begin
  Result := ExpandConstant('{localappdata}\LiteNotebook');
end;

function IsInside(Dir, Root: String): Boolean;
begin
  Dir := AddBackslash(Lowercase(Dir));
  Root := AddBackslash(Lowercase(Root));
  Result := Copy(Dir, 1, Length(Root)) = Root;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and IsInside(WizardDirValue, DataFolder()) then
  begin
    SuppressibleMsgBox('Please choose another folder.' + #13#10#13#10 + DataFolder() +
      ' is where LiteNotebook keeps your notebooks; the program must be installed elsewhere.',
      mbError, MB_OK, IDOK);
    Result := False;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if IsInside(WizardDirValue, DataFolder()) then
    Result := 'The program folder must not be inside ' + DataFolder() + ' (your notebooks).';
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and (not UninstallSilent) then
    MsgBox('LiteNotebook was removed.' + #13#10#13#10 +
      'Your notebooks, PDFs, notes, chats, backups, downloaded models and AI components were kept in:' + #13#10 +
      DataFolder() + #13#10#13#10 +
      'Delete that folder yourself only if you no longer need it (make a backup first).',
      mbInformation, MB_OK);
end;
