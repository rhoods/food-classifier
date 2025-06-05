#!/bin/bash
#
#echo "🔍 Vérification de Python3..."
#if ! command -v python3 &> /dev/null; then
#    echo "❌ python3 n'est pas installé. Installation en cours..."
#        sudo apt update && sudo apt install python3 -y
#        fi
#
#        echo "✅ python3 est présent : $(python3 --version)"
#
#        echo "🔄 Vérification de l'existence de 'python'..."
#        if ! command -v python &> /dev/null; then
#            echo "➕ Création du lien symbolique python -> python3"
#                sudo ln -s /usr/bin/python3 /usr/bin/python
#                else
#                    echo "✅ Le binaire 'python' est déjà disponible."
#                    fi
#
#                    echo "🛠️ Configuration de Poetry pour utiliser python3..."
#                    poetry env use python3
#
#                    echo "🧼 Suppression de l'ancien .venv si nécessaire..."
#                    rm -rf .venv
#
#                    echo "📦 Réinstallation propre des dépendances avec Poetry..."
#                    poetry install
#
#                    echo "✅ Terminé ! Tu peux maintenant activer ton environnement avec :"
#                    echo "   poetry shell"
#                    
