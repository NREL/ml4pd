"""
Module contains class Registry and a registry object that
keeps track of all initialized objects. To be used by flowsheet
to plot & pass data through graph structure.
"""

__all__ = ["registry"]


class Registry:
    """
    Class used to keep track of all initialized objects.

    Attributes:
        all_streams (dict): dictionary of all streams initialized.
        all_columns (dict): dicitonary of all columns initiliazed.

    """

    def __init__(self):
        self.all_streams = {}
        self.all_columns = {}

    def add_element(self, element):
        """Add element to all streams/columns based on its type."""
        if element.base_type == "stream":
            if element.object_id in self.all_streams:
                raise KeyError(f"{element.object_id} already taken.")
            self.all_streams[element.object_id] = element
        elif element.base_type == "unit-op":
            if element.object_id in self.all_columns:
                raise KeyError(f"{element.object_id} already taken.")
            self.all_columns[element.object_id] = element

    def remove_element(self, object_id: str):
        """Remove item given its id from all streams/columns."""
        if object_id in self.all_streams:
            del self.all_streams[object_id]
        elif object_id in self.all_columns:
            del self.all_columns[object_id]
        else:
            raise KeyError("id not found.")

    def clear_data(self):
        """Remove all streams & columns from registry."""
        self.all_streams = {}
        self.all_columns = {}

    def get_element(self, object_id: str):
        """Return object that corresponds to the id."""
        if object_id in self.all_streams:
            return_item = self.all_streams[object_id]
        elif object_id in self.all_columns:
            return_item = self.all_columns[object_id]
        else:
            raise KeyError("id not found.")

        return return_item

    def get_all_streams(self):
        """Returns dictionary of all streams."""
        return self.all_streams

    def get_all_columns(self):
        """Returns dictionary of all columns."""
        return self.all_columns


registry = Registry()
