import os
import fnmatch
from tkinter import filedialog, Tk

class FileSystemEntity:
    def __init__(self, path):
        self.path = path

class File(FileSystemEntity):
    def __init__(self, path):
        super().__init__(path)

class Directory(FileSystemEntity):
    def __init__(self, path, gitignore_patterns=None):
        super().__init__(path)
        self.contents = []
        self.gitignore_patterns = gitignore_patterns or []
        self.scan_directory()

    def scan_directory(self):
        for entry in os.listdir(self.path):
            if entry.startswith('.') or entry in ['node_modules', 'dist', 'build']:  # Skip hidden and common non-essential directories
                continue
            full_path = os.path.join(self.path, entry)
            if any(fnmatch.fnmatch(entry, pattern) for pattern in self.gitignore_patterns):
                continue  # Skip files/directories matched by .gitignore patterns
            if os.path.isdir(full_path):
                self.contents.append(Directory(full_path, self.gitignore_patterns))
            elif entry.endswith(('.js', '.html', '.css', '.py', '.jsx', '.json', '.md')):
                self.contents.append(File(full_path))  # Only include certain file types

    def print_structure(self, indent=0):
        print('  ' * indent + os.path.basename(self.path) + '/')
        for entity in self.contents:
            if isinstance(entity, Directory):
                entity.print_structure(indent + 1)
            else:
                print('  ' * (indent + 1) + os.path.basename(entity.path))

def parse_gitignore(root_path):
    gitignore_path = os.path.join(root_path, '.gitignore')
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, 'r') as file:
            return [line.strip() for line in file.readlines() if line.strip() and not line.startswith('#')]
    return []

def main():
    root = Tk()
    root.withdraw()
    selected_directory = filedialog.askdirectory()
    if selected_directory:
        gitignore_patterns = parse_gitignore(selected_directory)
        directory = Directory(selected_directory, gitignore_patterns)
        directory.print_structure()
    else:
        print("No directory selected.")

if __name__ == "__main__":
    main()