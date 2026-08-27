from mobpredict.networks.mhsa import TransEncoder
from mobpredict.networks.rnns import RNNs

__all__ = ["TransEncoder", "RNNs", "MambaEncoder"]


def __getattr__(name):
    """Load the optional Mamba implementation only when it is requested."""
    if name == "MambaEncoder":
        from mobpredict.networks.mamba import MambaEncoder

        return MambaEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
