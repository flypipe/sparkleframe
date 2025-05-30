from enum import Enum


class Engine(Enum):
    POLARS = ("polarsdf", "Polars")

    @property
    def module(self):
        return self.value[0]

    @property
    def class_prefix(self):
        return self.value[1]

    def clean_class_name(self, klass):
        return klass.replace(self.class_prefix, "") if klass.startswith(self.class_prefix) else klass
