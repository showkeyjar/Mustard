Set shell = CreateObject("WScript.Shell")
repoRoot = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
command = "cmd /c cd /d """ & repoRoot & """ && pythonw -m scripts.claw_team_control run"
shell.Run command, 0, False
