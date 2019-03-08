"""Exception classes specific to karabo_data."""


class SourceNameError(KeyError):
    def __init__(self, source, run=True):
        self.source = source
        self.run = run

    def __str__(self):
        run_file = 'run' if self.run else 'file'
        return (
            "This {0} has no source named {1!r}.\n"
            "See {0}.all_sources for available sources.".format(run_file, self.source)
        )


class PropertyNameError(KeyError):
    def __init__(self, prop, source):
        self.prop = prop
        self.source = source

    def __str__(self):
        return "No property {!r} for source {!r}".format(self.prop, self.source)
