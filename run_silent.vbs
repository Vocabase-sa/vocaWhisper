' =============================================
' VocaWhisper - Lanceur silencieux (admin)
' =============================================
' Lance le programme SANS ouvrir de fenetre CMD.
' S'auto-eleve en administrateur pour que les
' raccourcis clavier fonctionnent meme quand une
' fenetre admin est au premier plan (ex: VS Code).
' Double-cliquer sur ce fichier pour demarrer.

Set fso = CreateObject("Scripting.FileSystemObject")

' Chemin du script (meme dossier que ce .vbs)
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = scriptDir & "\venv\Scripts\pythonw.exe"
mainScript = scriptDir & "\whisper_dictation.py"

' Verifier que pythonw.exe existe
If Not fso.FileExists(pythonExe) Then
    MsgBox "pythonw.exe introuvable :" & vbCrLf & pythonExe & vbCrLf & vbCrLf & "Lancez d'abord install_windows.bat pour creer le venv.", vbCritical, "VocaWhisper"
    WScript.Quit
End If

' Auto-elevation en administrateur si pas deja admin
If WScript.Arguments.Count = 0 Then
    ' Relancer ce script en admin via ShellExecute "runas"
    CreateObject("Shell.Application").ShellExecute "wscript.exe", """" & WScript.ScriptFullName & """ /elevated", "", "runas", 0
    WScript.Quit
End If

' Delai au demarrage Windows pour laisser le GPU et l'audio s'initialiser
WScript.Sleep 5000

' Lancer en mode invisible (0 = hidden, False = ne pas attendre)
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """" & pythonExe & """ """ & mainScript & """", 0, False
