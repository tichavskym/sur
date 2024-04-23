import argparse
import numpy as np

# The function for parsing the user arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the models weights')    
    parser.add_argument('--gt', type=str, required=True, help='Path to the file containing the ground truth.')
    parser.add_argument('--ours', type=str, required=True, help='Path to the file containing our evaluationt.')    
    return parser.parse_args()

def load_data_ground_truth(file_path):
    parsed_data = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split the line into parts.
            parts = line.strip().split()
            
            if len(parts) >= 2:
                file_name = parts[0]
                gt = int(parts[1])

                parsed_data.append((file_name, gt))
            else:
                print(f"Error: invalid line format: {line}")

    # Convert the list of tuples to a structured NumPy array.
    dtype = [('file_name', '<U20'), ('gt', int)]
    parsed_array = np.array(parsed_data, dtype=dtype)

    return parsed_array

# Function for loading the data from the file in the required format.
def load_data_models(file_path):
    parsed_data = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split the line into parts.
            parts = line.strip().split()
            
            if len(parts) >= 3:
                file_name = parts[0]
                prob = float(parts[1])
                pred = int(parts[2])

                parsed_data.append((file_name, prob, pred))
            else:
                print(f"Error: invalid line format: {line}")
                break

    # Convert the list of tuples to a structured NumPy array.
    dtype = [('file_name', '<U20'), ('prob', float), ('pred', int)]
    parsed_array = np.array(parsed_data, dtype=dtype)

    return parsed_array

def eval(gt_data, ours_data):
    total = 0
    correct = 0

    for gt, ours in zip(gt_data, ours_data):
        # Check whether the files have the same file on the line.
        if gt['file_name'] == ours['file_name']:
            gt_pred = gt['gt']
            ours_pred = ours['pred']

            total += 1            

            if gt_pred == ours_pred:
                correct += 1
        else:
            print(f"Error: invalid line format during training was detected.")
            break
    
    return correct / total

if __name__ == "__main__":
    args = parse_arguments()

    gt_data = load_data_ground_truth(args.gt)    
    ours_data = load_data_models(args.ours)
    
    acc = eval(gt_data, ours_data)

    print("Our model accuracy: ", acc)
