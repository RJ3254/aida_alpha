"""Module that installs and loads Gemma 3n model."""
from __future__ import annotations

import subprocess
import sys


def install_gemma() -> None:
    """Install the Gemma library."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gemma==3.1.0"])


def load_gemma_3n():
    """Load Gemma 3n model.

    This function assumes that the gemma library is installed and all required
    dependencies are available. It demonstrates how to set up the Gemma 3n
    model for a simple chat prompt.
    """
    from gemma import gm
    # The Gemma 3n models live in a submodule that needs to be imported
    # explicitly before instantiation.
    from gemma.gm.nn import gemma3n

    model = gemma3n.Gemma3n_E4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3N_E4B_IT)
    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        multi_turn=True,
    )
    return sampler


if __name__ == "__main__":
    install_gemma()
    sampler = load_gemma_3n()
    print(sampler.chat("What's the capital of France?"))
