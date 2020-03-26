#!/bin/usr/env sh

echo "Full computation of BLI task for EN-ES, ES-EN, EN-FR, FR-EN, EN-DE, DE-EN, EN-RU, RU-EN "

#Available dico: German, Spanish, French, Italian, Portuguese


# ES-EN Induction using italian
sh bli.sh es it en

# DE-EN Induction using italien
sh bli.sh de it en

# FR-EN Induction using italian
sh bli.sh fr it en


# ES-EN Induction using deutsh
sh bli.sh es de en

# FR-EN Induction using deutsh
sh bli.sh fr de en


# ES-EN Induction using french
sh bli.sh es fr en

# DE-EN Induction using french
sh bli.sh de fr en


# FR-EN Induction using spanish
sh bli.sh fr es en

# DE-EN Induction using spanish
sh bli.sh de es en