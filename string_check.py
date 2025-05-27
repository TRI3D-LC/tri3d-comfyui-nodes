import os

class TRI3D_StringContains:
    """
    ComfyUI node that checks if a specified string exists within another string.
    Performs case-insensitive comparison by converting all text to lowercase.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": True}),
                "search_string": ("STRING", {"default": "", "multiline": False}),
            },
        }
    
    FUNCTION = "run"
    RETURN_TYPES = ("BOOLEAN",)
    CATEGORY = "TRI3D"
    
    def run(self, input_string, search_string):
        # Convert both strings to lowercase for case-insensitive comparison
        input_lower = input_string.lower()
        search_lower = search_string.lower()
        
        # Check if search string exists in input string
        contains = search_lower in input_lower
        
        return (contains,)

# # Node registration for ComfyUI
# NODE_CLASS_MAPPINGS = {
#     "TRI3D_StringContains": TRI3D_StringContains
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "TRI3D_StringContains": "TRI3D String Contains"
# }
