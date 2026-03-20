Set shell = CreateObject("WScript.Shell")
repoRoot = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
command = "cmd /c cd /d """ & repoRoot & """ && pythonw -m scripts.desktop_agent_control launch"
shell.Run command, 0, False
