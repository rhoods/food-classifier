FROM python:3.10-slim

# Étape 1 : Dépendances système de base
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Étape 2 : Installer Poetry
ENV POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Étape 3 : Définir le dossier de travail
WORKDIR /app

# Étape 4 : Copier les fichiers du projet
COPY pyproject.toml poetry.lock* /app/

# Étape 5 : Installer les dépendances (hors editable install)
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

# Étape 6 : Copier le reste du code
COPY . /app

# Étape 7 : Commande par défaut
CMD ["python", "food_classifier/train.py"]

