import os

# Configuration: Add folders or files you want to ignore
IGNORE_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build'}
IGNORE_FILES = {'.DS_Store', 'package-lock.json', 'yarn.lock', 'bundle.txt', '.md'}
ALLOWED_EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.json', '.txt', '.cpp', '.hpp'}

def bundle_repo(output_file='bundle.txt'):
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk('.'):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                ext = os.path.splitext(file)[1]
                if ext in ALLOWED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, '.')
                    
                    f.write(f"\n{'='*50}\n")
                    f.write(f"FILE: {relative_path}\n")
                    f.write(f"{'='*50}\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as code_file:
                            f.write(code_file.read())
                    except Exception as e:
                        f.write(f"[Error reading file: {e}]")
                    f.write("\n")

    print(f"Done! Your context is ready in {output_file}")

if __name__ == "__main__":
    bundle_repo()

