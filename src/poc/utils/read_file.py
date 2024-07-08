# import: standard
import pathlib

# import: external
import yaml


def read_yaml(filepath: str) -> dict:
    """
    Read a YAML file and return its contents as a Python dictionary.

    Args:
        filepath: Path to the YAML file to read.

    Returns:
        Dictionary containing the YAML file contents.

    Raises:
        ValueError: If the file is not a YAML file.
    """
    # Define valid YAML file extensions
    yaml_file_extension = (".yaml", ".yml")

    # Create a pathlib.Path object from the filepath
    path_obj = pathlib.Path(filepath)

    # Check if the file extension is one of the valid YAML extensions
    if not path_obj.suffix.lower().endswith(yaml_file_extension):
        raise ValueError(f"'{filepath}' is not a YAML file.")

    # Read the contents of the file as text
    content = path_obj.read_text()

    # Safely load YAML content into a Python dictionary
    return yaml.safe_load(content)


def is_image_file(image_filepath: str) -> bool:
    """
    Check if a given file path corresponds to an image file.

    Args:
        image_filepath (str): Path to the file to check.

    Returns:
        bool: True if the file at image_filepath has an image extension, False otherwise.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    file_extension = pathlib.Path(image_filepath).suffix.lower()  # Get the lowercase file extension
    return file_extension in image_extensions
