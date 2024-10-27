import spacy.cli
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForTokenClassification
import numpy as np
import re
import spacy

class MentalHealthPhraseExtractor:
    def __init__(self):
        # Load Clinical BERT for medical phrase extraction
        self.clinical_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.clinical_model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # Load medical NER pipeline for symptom detection
        self.ner_pipeline = pipeline(
            "ner",
            model="samrawal/bert-base-uncased_clinical-ner",
            aggregation_strategy="simple"
        )

        # Load spaCy NLP model for POS tagging
        self.nlp = spacy.load("en_core_web_sm")

        # Regex for pronouns, contractions, conjunctions
        self.pronoun_pattern = re.compile(
            r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|"
            r"yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|"
            r"itself|they|them|their|theirs|themselves|im|i'm|i've|i'll|you've|"
            r"you'll|we've|we'll|they've|they'll|he's|she's|it'll|it'd|we'd|they'd|i'd|"
            r"'ve|'m|'ll|'re|'d|and|or|but|also|so|because|however|therefore|thus)\b", 
            re.IGNORECASE
        )

        # Regex to split at verbs/adverbs
        self.verbs_and_modifiers = re.compile(
            r"\b(?:feel|feeling|be|been|am|is|was|are|were|have|has|had|very|really|"
            r"seem|seemed|find|seems|can|could|would|should|do|did|does|why|how|what|when|where|who)\b", 
            re.IGNORECASE
        )

        # Final regex to clean orphaned artifacts like "'t"
        self.orphan_pattern = re.compile(r"\b't\b|'[a-zA-Z]\b")

        self.used_spans = set()

    def extract_phrases(self, text):
        """Extract mental health related phrases using clinical models."""
        self.used_spans.clear()
        phrases = []

        # 1. Extract clinical entities
        entities = self._extract_clinical_entities(text)
        phrases.extend(entities)

        # 2. Extract relevant phrases using sliding window
        chunks = self._get_text_chunks(text)
        for chunk in chunks:
            if chunk['confidence'] > 0.4:  # Only keep significant phrases
                if not self._is_overlapping(chunk['start'], chunk['end']):
                    self.used_spans.add((chunk['start'], chunk['end']))
                    phrases.append(chunk)

        # Sort phrases by position in text
        phrases.sort(key=lambda x: x['start'])

        # Merge adjacent phrases, clean them, and split further
        merged_phrases = self._merge_phrases(phrases)

        return merged_phrases

    def _extract_clinical_entities(self, text):
        """Extract clinical entities using medical NER."""
        entities = self.ner_pipeline(text)
        relevant_entities = []

        for entity in entities:
            if entity['score'] > 0.4:
                start = text.find(entity['word'])
                if start != -1 and not self._is_overlapping(start, start + len(entity['word'])):
                    self.used_spans.add((start, start + len(entity['word'])))
                    relevant_entities.append({
                        'phrase': entity['word'],
                        'type': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': start,
                        'end': start + len(entity['word'])
                    })

        return relevant_entities

    def _get_text_chunks(self, text, window_size=5):
        """Get text chunks using sliding window."""
        words = text.split()
        chunks = []

        for i in range(len(words)):
            for j in range(i + 1, min(i + window_size, len(words) + 1)):
                chunk = ' '.join(words[i:j])
                start = text.find(chunk)
                if start != -1:
                    inputs = self.clinical_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = self.clinical_model(**inputs)
                        predictions = torch.sigmoid(outputs.logits)
                        confidence = predictions.mean().item()
                        if confidence > 0.4:
                            chunks.append({
                                'phrase': chunk,
                                'confidence': confidence,
                                'start': start,
                                'end': start + len(chunk)
                            })

        return chunks

    def _is_overlapping(self, start, end):
        """Check if span overlaps with any used spans."""
        return any(start < used_end and end > used_start for used_start, used_end in self.used_spans)

    def _merge_phrases(self, phrases, max_gap=2):
        """Merge overlapping phrases and clean further."""
        if not phrases:
            return []

        merged = [phrases[0]]
        for current in phrases[1:]:
            previous = merged[-1]
            if current['start'] <= previous['end'] + max_gap:
                merged[-1] = {
                    'phrase': f"{previous['phrase']} {current['phrase']}",
                    'start': previous['start'],
                    'end': current['end'],
                    'confidence': max(previous['confidence'], current['confidence']),
                    'type': previous.get('type', '') or current.get('type', '')
                }
            else:
                merged.append(current)

        # Clean and split further
        final_phrases = []
        for phrase in merged:
            cleaned_phrases = self._split_atomic(phrase['phrase'])
            final_phrases.extend(cleaned_phrases)

        return final_phrases

    def _split_atomic(self, phrase):
        """Split phrase into smaller parts and filter irrelevant ones."""
        cleaned_phrase = self.pronoun_pattern.sub("", phrase)
        split_phrases = re.split(self.verbs_and_modifiers, cleaned_phrase)
        atomic_phrases = [p.strip() for p in split_phrases if self._is_relevant(p.strip())]
        return [self._clean_orphan(p) for p in atomic_phrases]

    def _clean_orphan(self, phrase):
        """Remove orphaned contractions like 't."""
        return self.orphan_pattern.sub("", phrase).strip()

    def _is_relevant(self, word):
        """Keep only relevant words (nouns/adjectives)."""
        doc = self.nlp(word)
        return any(token.pos_ in {"NOUN", "ADJ"} for token in doc)

def remove_apostrophe_parts(phrases):
    # Pattern to match the apostrophe followed by a letter with no letter before
    pattern = r"(?<!\w)(’[a-zA-Z])"
    modified_phrases = []

    for phrase in phrases:
        # Replace the matched pattern with an empty string (remove it)
        modified_phrase = re.sub(pattern, '', phrase)
        modified_phrases.append(modified_phrase.strip())

    return modified_phrases

def getKeyPhrases(text):
    extractor = MentalHealthPhraseExtractor()
    phrases = extractor.extract_phrases(text)
    phrases = remove_apostrophe_parts(phrases)
    return phrases

def main():
    extractor = MentalHealthPhraseExtractor()

    sample_text = """
    I've been feeling really down lately and can't seem to focus on anything. 
    My sleep has been terrible, waking up multiple times and feeling exhausted. 
    I'm constantly worried about work and find myself avoiding social situations. 
    My appetite has decreased and I feel tense most of the time. 
    Sometimes my thoughts race and I can't calm down.
    """

    sample_text = """
    Ugh, woke up and I feel like I’m in a pressure cooker. 
    Seriously, can’t even get out of bed without feeling like the whole world is about to collapse on me. 
    Just the thought of going outside makes me want to scream. 
    Why is it so hard to breathe? Why does everything feel like a huge deal? I just want to hide!
    """

    sample_text = """
    I dont understand My husband of 8 years said he wants a divorce. 
    He recently gave up drinking and has never developed healthy coping skills. 
    I am trying to be supportive, and he told me just a week ago how I was the love of his life.  
    If we fought a ton, didn't have a good sex life, and didn't care, I'd get it. 
    I don't understand why he wants to throw away our marriage?
    """

    phrases = extractor.extract_phrases(sample_text)
    phrases = remove_apostrophe_parts(phrases)
    print("Extracted Mental Health Indicators:")
    for phrase in phrases:
        print(f"- {phrase}")

    print(phrases)

if __name__ == "__main__":
    main()
