#!/bin/bash
#
## Récupérer le chemin de l'env Poetry
#VENV_PATH=$(poetry env info --path 2>/dev/null)
#
#if [ -z "$VENV_PATH" ]; then
#  echo "❌ Aucun environnement Poetry trouvé. As-tu bien fait 'poetry install' ?"
#    exit 1
#    fi
#
#    # Activer l'env
#    echo "🔄 Activation de l'environnement virtuel Poetry situé dans : $VENV_PATH"
#    source "$VENV_PATH/bin/activate"
#
#    # Afficher les chemins utilisés
#    echo "✅ Environnement activé."
#    echo "Python utilisé : $(which python)"
#    echo "Pip utilisé : $(which pip)"
