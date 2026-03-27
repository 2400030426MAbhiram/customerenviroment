import random
import uuid
from openenv.core.env_server import Environment
from .models import WordGameAction, WordGameObservation, WordGameState

WORDS = ["python", "neural", "tensor", "matrix", "vector",
         "kernel", "lambda", "signal", "binary", "cipher"]

class WordGameEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True  # Allow multiple simultaneous clients

    MAX_ATTEMPTS = 10

    def __init__(self):
        self._state = WordGameState()
        self._target = ""
        self._guessed = set()
        self._remaining = self.MAX_ATTEMPTS

    def reset(self, seed=None, episode_id=None, **kwargs) -> WordGameObservation:
        self._target = random.choice(WORDS)
        self._guessed = set()
        self._remaining = self.MAX_ATTEMPTS
        self._state = WordGameState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            target_word=self._target,
            max_attempts=self.MAX_ATTEMPTS,
        )
        return WordGameObservation(
            done=False,
            reward=None,
            masked_word=self._mask(),
            guessed_letters=[],
            attempts_remaining=self._remaining,
            message=f"Guess letters in a {len(self._target)}-letter word!",
        )

    def step(self, action: WordGameAction, timeout_s=None, **kwargs) -> WordGameObservation:
        letter = action.guess.lower().strip()
        self._state.step_count += 1
        self._guessed.add(letter)

        if letter in self._target:
            message = f"'{letter}' is in the word!"
        else:
            self._remaining -= 1
            message = f"'{letter}' is not in the word."

        # Check win/lose
        masked = self._mask()
        won = "_" not in masked
        lost = self._remaining <= 0
        done = won or lost

        if won:
            reward = 1.0
            message = f"You got it! The word was '{self._target}'."
        elif lost:
            reward = 0.0
            message = f"Out of attempts. The word was '{self._target}'."
        else:
            reward = 0.0

        return WordGameObservation(
            done=done,
            reward=reward,
            masked_word=masked,
            guessed_letters=sorted(self._guessed),
            attempts_remaining=self._remaining,
            message=message,
        )

    @property
    def state(self) -> WordGameState:
        return self._state

    def _mask(self) -> str:
        return "".join(c if c in self._guessed else "_" for c in self._target)