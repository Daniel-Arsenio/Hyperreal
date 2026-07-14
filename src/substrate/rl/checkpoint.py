import os
from typing import Any

import orbax.checkpoint as ocp


def save_checkpoint(directory: str, params: Any) -> None:
    directory = os.path.abspath(directory)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(directory, params, force=True)
    checkpointer.wait_until_finished()


def load_checkpoint(directory: str, target: Any) -> Any:
    directory = os.path.abspath(directory)
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer.restore(directory, target=target)
