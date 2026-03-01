' =============================================
' Whisper Dictation - Lanceur silencieux
' =============================================
' Lance le programme SANS ouvrir de fenetre CMD.
' Double-cliquer sur ce fichier pour demarrer.

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Chemin du script (meme dossier que ce .vbs)
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = scriptDir & "\venv\Scripts\pythonw.exe"
mainScript = scriptDir & "\whisper_dictation.py"

' Verifier que pythonw.exe existe
If Not fso.FileExists(pythonExe) Then
    MsgBox "pythonw.exe introuvable :" & vbCrLf & pythonExe & vbCrLf & vbCrLf & "Lance d'abord install.bat pour creer le venv.", vbCritical, "Whisper Dictation"
    WScript.Quit
End If

' Delai au demarrage Windows pour laisser le GPU et l'audio s'initialiser
WScript.Sleep 5000

' Lancer en mode invisible (0 = hidden, False = ne pas attendre)
WshShell.Run """" & pythonExe & """ """ & mainScript & """", 0, False
