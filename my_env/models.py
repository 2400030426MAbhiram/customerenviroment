from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

# Action, Observation, State are Pydantic BaseModel subclasses —
# no @dataclass decorator needed; define fields directly as class attributes.

class WordGameAction(Action):
    guess: str  # The player's guessed letter

class WordGameObservation(Observation):
    # done: bool and reward: Optional[float] are already in Observation base
    masked_word: str           # e.g., "h_ll_"
    guessed_letters: List[str] # Letters tried so far
    attempts_remaining: int
    message: str               # Feedback message

class WordGameState(State):
    # episode_id: Optional[str] and step_count: int are already in State base
    target_word: str = ""
    max_attempts: int = 10