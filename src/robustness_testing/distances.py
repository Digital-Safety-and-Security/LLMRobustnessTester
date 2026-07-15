import editdistance

from robustness_testing import Distance
from robustness_testing.core import RobustnessObject

class EditDistance(Distance):
    def distance(self, left: RobustnessObject[str], right: RobustnessObject[str]) -> float:
        """Return the edit distance between two strings."""
        return float(editdistance.eval(left.value, right.value))

class WordEditDistance(Distance):
    def distance(self, left: RobustnessObject[str], right: RobustnessObject[str]) -> float:
        """Return the edit distance between two strings, computed at the word level."""
        left_words = left.value.lower().split()
        right_words = right.value.lower().split()
        return float(editdistance.eval(left_words, right_words))
