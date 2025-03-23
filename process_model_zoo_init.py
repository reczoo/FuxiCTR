import os
import re
from pathlib import Path

def process_model_zoo_init(model_zoo_dir):
    root_init = model_zoo_dir / "__init__.py"
    model_imports = []
    
    # Parse all import lines from root __init__.py
    with open(root_init, "r") as f:
        for line in f:
            if line.strip().startswith(("#", "\n")):
                continue  # Skip comments and empty lines
            match = re.match(r"^from \.((?:\w+\.?)+) import (\w+(?:,\s*\w+)*)", line)
            if match:
                import_path = match.group(1)
                models = [m.strip() for m in match.group(2).split(",")]
                model_imports.append((import_path, models))
    
    # Process each import path
    for import_path, models in model_imports:
        path_parts = import_path.split(".")
        current_dir = model_zoo_dir
        
        # Create directory structure and empty __init__.py files
        for part in path_parts:
            current_dir = current_dir / part
            if not current_dir.exists():
                current_dir.mkdir()
            init_file = current_dir / "__init__.py"
            if not init_file.exists():
                init_file.touch()
        
        # Add model imports to deepest __init__.py
        deepest_init = current_dir / "__init__.py"
        with open(deepest_init, "a") as f:
            for model in models:
                import_line = f"from .{model} import *\n"
                # Check if import already exists
                with open(deepest_init, "r") as rf:
                    existing_content = rf.read()
                if import_line not in existing_content:
                    f.write(import_line)

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent  # Adjust based on your structure
    model_zoo_dir = project_root / "FuxiCTR" / "model_zoo"
    process_model_zoo_init(model_zoo_dir)