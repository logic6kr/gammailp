import os 
import yaml
import sys

def read_yaml(file_path):
    """Read a YAML file and return its content."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    task_name = sys.argv[1] if len(sys.argv) > 1 else 'default_task'
    default_yaml = '../config/adjr.yaml'
    output_dir = f'../config/{task_name}.yaml'
    yaml_current = read_yaml(default_yaml)
    print(f"Using default YAML: {default_yaml}")

    # # Write the YAML data to a file
    # with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
    #     yaml.dump(data, file, default_flow_style=False)

    # print(f"YAML configuration has been written to {os.path.join(output_dir, 'config.yaml')}")
