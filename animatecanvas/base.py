"""Small inference base; no registry or training framework dependency."""

class BasePipeline:
    def __init__(self, bundle):
        self.bundle = bundle
        self.bundle.eval()
