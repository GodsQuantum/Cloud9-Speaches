Unicode true
!include "MUI2.nsh"
!ifndef STAGING
  !error "Missing STAGING source folder"
!endif
!ifndef OUTPUT
  !error "Missing OUTPUT setup EXE path"
!endif

Name "Cloud9-Speaches"
OutFile "${OUTPUT}"
InstallDir "$LOCALAPPDATA\Programs\Cloud9-Speaches"
InstallDirRegKey HKCU "Software\GodsQuantum\Cloud9-Speaches" "InstallPath"
RequestExecutionLevel user
SetCompressor /SOLID lzma
ShowInstDetails show
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Install" SEC_MAIN
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  File /r "${STAGING}\*.*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\Cloud9-Speaches"
  CreateShortcut "$SMPROGRAMS\Cloud9-Speaches\Cloud9-Speaches.lnk" "$INSTDIR\Cloud9-Speaches.exe"
  CreateShortcut "$SMPROGRAMS\Cloud9-Speaches\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\GodsQuantum\Cloud9-Speaches" "InstallPath" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\GodsQuantum-Cloud9-Speaches" "DisplayName" "Cloud9-Speaches"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\GodsQuantum-Cloud9-Speaches" "DisplayVersion" "0.1.0"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\GodsQuantum-Cloud9-Speaches" "UninstallString" '"$INSTDIR\Uninstall.exe"'
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  Delete "$SMPROGRAMS\Cloud9-Speaches\Cloud9-Speaches.lnk"
  Delete "$SMPROGRAMS\Cloud9-Speaches\Uninstall.lnk"
  RMDir "$SMPROGRAMS\Cloud9-Speaches"
  RMDir /r "$INSTDIR\_internal"
  RMDir /r "$INSTDIR\bin"
  Delete "$INSTDIR\Cloud9-Speaches.exe"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\GodsQuantum-Cloud9-Speaches"
  DeleteRegKey HKCU "Software\GodsQuantum\Cloud9-Speaches"
  ; Preserve per-user models/settings under LOCALAPPDATA Cloud9-Speaches.
SectionEnd
