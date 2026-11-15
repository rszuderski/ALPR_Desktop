"""Private evidence is opt-in; never infer an author's home folder."""
import os
from pathlib import Path

def private_test_path(relative):
    root = os.environ.get('ALPR_TEST_DATA_DIR')
    return (Path(root) if root else Path(__file__).parent / 'private_fixtures') / relative
