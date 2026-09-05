-- =============================================================================
-- Script 1 : Ajout des colonnes complete_transcription et success-flag
-- Table    : calls3
-- Base     : vocabase (production)
-- =============================================================================
-- A executer UNE SEULE FOIS sur la base de production.
-- =============================================================================

ALTER TABLE calls3
  ADD COLUMN IF NOT EXISTS `complete_transcription` TEXT DEFAULT NULL;

ALTER TABLE calls3
  ADD COLUMN IF NOT EXISTS `success-flag` INT(11) DEFAULT NULL;
