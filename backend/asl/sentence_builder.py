"""
Sentence Builder & Prediction Stabilization
============================================
Handles temporal prediction smoothing, hold verification, duplicate suppression,
progressive letter accumulation, and sentence editing.
"""

import time
from typing import List, Optional, Tuple

from .config import (
    FS_CONFIDENCE_THRESHOLD,
    FS_HOLD_FRAMES,
    LETTER_PAUSE_SECONDS,
    WORD_PAUSE_SECONDS,
    MAX_SENTENCE_WORDS,
)


class SentenceBuilder:
    """
    Maintains progressive sentence state for a single ASL recognition session.
    """

    def __init__(
        self,
        hold_frames: int = FS_HOLD_FRAMES,
        letter_pause: float = LETTER_PAUSE_SECONDS,
        word_pause: float = WORD_PAUSE_SECONDS,
    ):
        self.hold_frames = hold_frames
        self.letter_pause = letter_pause
        self.word_pause = word_pause

        self.sentence_words: List[str] = []
        self.current_letters: List[str] = []

        # Fingerspelling stabilization state
        self.prev_letter: str = ""
        self.hold_count: int = 0
        self.last_letter_time: float = 0.0
        self.last_hand_time: float = time.time()

        # Word mode state
        self.current_word: str = ""
        self.last_accepted_prediction: str = ""

    def update_hand_presence(self, has_hand: bool, now: Optional[float] = None):
        """Update hand presence timestamp and check auto-space on hand pause."""
        if now is None:
            now = time.time()

        if has_hand:
            self.last_hand_time = now
        else:
            # If no hand has been detected for longer than word_pause, commit current letters
            if (now - self.last_hand_time) > self.word_pause and self.current_letters:
                self.sentence_words.append("".join(self.current_letters))
                self.current_letters = []

    def feed_letter(
        self, letter: str, confidence: float, now: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Feed an alphabet model prediction frame.
        Returns:
            (accepted: bool, hold_progress: float)
        """
        if now is None:
            now = time.time()

        letter_clean = letter.strip()
        if confidence < FS_CONFIDENCE_THRESHOLD or letter_clean.lower() in ("nothing", ""):
            # Reset hold if prediction dropped or is nothing
            if letter_clean.lower() in ("nothing", ""):
                self.hold_count = 0
                self.prev_letter = ""
            return False, 0.0

        if letter_clean == self.prev_letter:
            self.hold_count += 1
        else:
            self.hold_count = 0
            self.prev_letter = letter_clean

        hold_progress = min(1.0, float(self.hold_count) / float(self.hold_frames))
        accepted = False

        if (
            self.hold_count >= self.hold_frames
            and (now - self.last_letter_time) > self.letter_pause
        ):
            action = letter_clean.lower()
            if action == "space":
                if self.current_letters:
                    self.sentence_words.append("".join(self.current_letters))
                    self.current_letters.clear()
                accepted = True
            elif action == "del":
                if self.current_letters:
                    self.current_letters.pop()
                elif self.sentence_words:
                    self.sentence_words.pop()
                accepted = True
            else:
                self.current_letters.append(letter_clean.upper())
                accepted = True

            self.last_letter_time = now
            self.hold_count = 0
            hold_progress = 0.0

        self._truncate_sentence()
        return accepted, hold_progress

    def feed_word(
        self, word: str, confidence: float, is_signing: bool = False
    ) -> bool:
        """
        Feed a word model prediction into the sentence builder when high confidence
        and motion indicates sign completion.
        """
        word_clean = word.strip()
        if not word_clean or word_clean == "?":
            return False

        if not is_signing and word_clean != self.current_word:
            self.current_word = word_clean
            self.sentence_words.append(word_clean)
            self.last_accepted_prediction = word_clean
            self._truncate_sentence()
            return True

        return False

    def space(self):
        """Commit current fingerspelled letters to sentence as a complete word."""
        if self.current_letters:
            self.sentence_words.append("".join(self.current_letters))
            self.current_letters.clear()
        self._truncate_sentence()

    def undo(self):
        """Remove the last accepted token (letter or word)."""
        if self.current_letters:
            self.current_letters.pop()
        elif self.sentence_words:
            self.sentence_words.pop()

    def clear(self):
        """Reset all sentence buffers."""
        self.sentence_words.clear()
        self.current_letters.clear()
        self.current_word = ""
        self.hold_count = 0
        self.prev_letter = ""

    def commit(self, word_override: Optional[str] = None):
        """Explicitly commit the current letters or word override."""
        if word_override:
            self.sentence_words.append(word_override)
        elif self.current_letters:
            self.sentence_words.append("".join(self.current_letters))
            self.current_letters.clear()
        elif self.current_word:
            self.sentence_words.append(self.current_word)
        self._truncate_sentence()

    def get_current_word(self) -> str:
        """Return the word currently being fingerspelled."""
        return "".join(self.current_letters)

    def get_sentence(self) -> str:
        """Return the full synthesized sentence."""
        words = list(self.sentence_words)
        if self.current_letters:
            words.append("".join(self.current_letters))
        return " ".join(words).strip()

    def _truncate_sentence(self):
        """Prevent unbounded sentence memory growth."""
        if len(self.sentence_words) > MAX_SENTENCE_WORDS:
            self.sentence_words = self.sentence_words[-MAX_SENTENCE_WORDS:]

