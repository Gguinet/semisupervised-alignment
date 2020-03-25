#!/bin/usr/env sh

echo "Full computation of BLI task for EN-ES, ES-EN, EN-FR, FR-EN, EN-DE, DE-EN, EN-RU, RU-EN "


# EN-ES Induction using italian
sh bli.sh en it es

# EN-ES Induction using Portuguese
sh bli.sh en pt es

# EN-ES Induction using French
sh bli.sh en fr es


# ES-EN Induction using italian
sh bli.sh es it en

# ES-EN Induction using deutsh
sh bli.sh es de en

# ES-EN Induction using french
sh bli.sh es fr en



# EN-FR Induction using spanish
sh bli.sh en es fr

# EN-FR Induction using deutsh
sh bli.sh en de fr

# EN-FR Induction using italian
sh bli.sh en it fr



# FR-EN Induction using deutsh
sh bli.sh fr de en

# FR-EN Induction using spanish
sh bli.sh fr es en

# FR-EN Induction using italian
sh bli.sh fr it en



# EN-DE Induction using dutch
sh bli.sh en nl de

# EN-DE Induction using danish
sh bli.sh en da de

# EN-DE Induction using polish
sh bli.sh en pl de

# EN-DE Induction using french
sh bli.sh en fr de


# dico de nl da pl existe pas 

# DE-EN Induction using spanish
sh bli.sh de es en

# DE-EN Induction using italien
sh bli.sh de it en

# DE-EN Induction using french
sh bli.sh de fr en



# EN-RU Induction using Polish
sh bli.sh en pl ru

# EN-RU Induction using Czech
sh bli.sh en cs ru

# EN-RU Induction using dutch
sh bli.sh en nl ru

# EN-RU Induction using german
sh bli.sh en de ru

#None of these exist

# RU-EN Induction using Polish
#sh bli.sh ru pl en

# RU-EN Induction using Czech
#sh bli.sh ru cs en

# RU-EN Induction using dutch
#sh bli.sh ru nl en

# RU-EN Induction using german
#sh bli.sh ru de en
