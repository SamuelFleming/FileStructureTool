import os
from tkinter import filedialog, Tk

class FileSystemEntity:
    def __init__(self, path):
        self.path = path

class File(FileSystemEntity):
    def __init__(self, path):
        super().__init__(path)

class Directory(FileSystemEntity):
    def __init__(self, path):
        super().__init__(path)
        self.contents = []
        self.scan_directory()

    def scan_directory(self):
        """Scans the directory and populates contents with File and Directory objects"""
        for entry in os.listdir(self.path):
            full_path = os.path.join(self.path, entry)
            if os.path.isdir(full_path):
                self.contents.append(Directory(full_path))
            else:
                self.contents.append(File(full_path))

    def print_structure(self, indent=0):
        """Prints the structure of the directory"""
        print('  ' * indent + os.path.basename(self.path) + '/')
        for entity in self.contents:
            if isinstance(entity, Directory):
                entity.print_structure(indent + 1)
            else:
                print('  ' * (indent + 1) + os.path.basename(entity.path))


def main():
    root = Tk()
    root.withdraw() # Hide the main window
    selected_directory = filedialog.askdirectory()
    if selected_directory:
        directory = Directory(selected_directory)
        directory.print_structure()
    else:
        print("No directory selected.")

if __name__ == "__main__":
    main()
