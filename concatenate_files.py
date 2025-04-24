import os
import fnmatch

def get_project_structure(exclude_dirs):
    structure = []
    for root, dirs, files in os.walk('.'):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        level = root.replace('./', '').count(os.sep)
        indent = '  ' * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        
        subindent = '  ' * (level + 1)
        for f in sorted(files):
            structure.append(f"{subindent}{f}")
    
    return '\n'.join(structure)

def concatenate_files(output_file='all_files.txt', 
                     extensions=('.py', '.cpp', '.h', '.txt', '.md'),
                     exclude_dirs=('__pycache__', 'build', '.git'),
                     include_paths=True):
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write project structure at the beginning
        outfile.write("PROJECT STRUCTURE\n")
        outfile.write("="*80 + "\n\n")
        outfile.write(get_project_structure(exclude_dirs))
        outfile.write("\n\n" + "="*80 + "\n\n")
        outfile.write("FILE CONTENTS\n")
        outfile.write("="*80 + "\n")
        
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith(extensions):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            if include_paths:
                                outfile.write(f"\n\n{'='*80}\n")
                                outfile.write(f"File: {filepath}\n")
                                outfile.write(f"{'='*80}\n\n")
                            content = infile.read()
                            outfile.write(content)
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

if __name__ == '__main__':
    concatenate_files() 