import os
from pathlib import Path

def cleanup_unmatched_files(img_dir, txt_dir):
    """
    Delete JPG files that don't have corresponding TXT files.
    
    Args:
        img_dir (str): Directory containing JPG files
        txt_dir (str): Directory containing TXT annotation files
    
    Returns:
        tuple: Number of files kept and deleted
    """
    # Convert paths to Path objects for easier handling
    img_path = Path(img_dir)
    txt_path = Path(txt_dir)
    
    # Get lists of files
    jpg_files = set(f.stem for f in img_path.glob('*.jpg'))
    txt_files = set(f.stem for f in txt_path.glob('*.txt'))
    
    # Find files to delete (jpg files without corresponding txt)
    files_to_delete = jpg_files - txt_files
    
    # Keep track of statistics
    deleted_count = 0
    kept_count = len(jpg_files - files_to_delete)
    
    # Delete unmatched files
    for filename in files_to_delete:
        file_to_delete = img_path / f"{filename}.jpg"
        try:
            file_to_delete.unlink()
            deleted_count += 1
            print(f"Deleted: {file_to_delete}")
        except Exception as e:
            print(f"Error deleting {file_to_delete}: {e}")
    
    return kept_count, deleted_count

def main():
    # Your specific paths
    img_directory = "/home/xavier/Robolab/ZionPrestonData"
    txt_directory = "/home/xavier/Robolab/ZionPrestonData/ZPAnnotations/obj_train_data"
    
    # Create backup before proceeding
    print("Creating backup of image directory...")
    from datetime import datetime
    import shutil
    
    backup_dir = f"{img_directory}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(img_directory, backup_dir)
    print(f"Backup created at: {backup_dir}")
    
    # Confirm with user
    print("\nWarning: This script will permanently delete JPG files without matching TXT files.")
    confirm = input("Do you want to proceed? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    # Process files
    print("\nProcessing files...")
    kept, deleted = cleanup_unmatched_files(img_directory, txt_directory)
    
    # Print summary
    print(f"\nOperation completed:")
    print(f"Files kept: {kept}")
    print(f"Files deleted: {deleted}")
    print(f"Total files processed: {kept + deleted}")

if __name__ == "__main__":
    main()