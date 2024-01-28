# README zur FH Technikum Wien TeX-Vorlage
Speichern Sie Ihre Bilder im Ordner PICs
Ändern Sie nichts an den Files im Ordner BASE

## Zitierstile
Sie haben in diesem Template die Wahl zwischen vordefinierten Zitierstilen von einzelnen Studiengängen der FH Technikum und den selbst zu modifizierenden LaTeX packages. Bitte sprechen Sie unbedingt vorab mit Ihrer/Ihrem Betreuer*in ob der Zitierstil in so einer Form passt. 

Für den Harvard-Stil verwenden Sie:
        \documentclass[Bachelor,BMR,english,fhCitStyle,HARVARD]{BASE/twbook}

Für den IEEE-Stil verwenden Sie:
        \documentclass[Bachelor,BMR,english,fhCitStyle,IEEE]{BASE/twbook}

Wenn Sie Stil-Anpassungen vornehmen wollen verwenden Sie:
        \documentclass[Bachelor,BMR,english]{BASE/twbook} 
und legen die entsprechenden Anpassungen in der Section 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Zitierstil zum selbst definieren
fest. Wir empfehlen dabei als Grundlage
        \usepackage[backend=biber, style=ieee]{biblatex}            % LaTeX definierter IEEE- Standard
oder 
        \usepackage[backend=biber, style=authoryear]{biblatex}     % LaTeX definierter Harvard-Standard
und legen Sie das File für Ihr Literaturverzeichnis fest mit
        \addbibresource{Literatur.bib}                              % Literatur-File definieren

Das Erstellen des Literatur-Verzeichnis machen Sie gegen Ende Ihres Dokuments mit
        \printbib{Literatur}      
für FH Technikum Wien Zitierstile, wobei {Literatur} das File Literatur.bib als Quelle festlegt und mit 
        \printbib
für selbst modifizierte Zitierstile, wobei das Literatur.bib schon oben als Quelle mit \addbibresource festgelegt wurde.

WICHTIG: Wenn Sie zwischen IEEE und HARVARD wechseln, bzw. sonstige Veränderungen in der Definition Ihres Zitierstils machen bitte die temporären Dateien (aux, bbl, ...) löschen bzw. in Overleaf Recompile from Scratch wählen.
