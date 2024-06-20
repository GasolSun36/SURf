import json
import argparse

def process_file(input_file, output_file):

    with open(input_file, 'r') as file:
        data = json.load(file)


    new_data = []


    for item in data:
        id = item['id']
        image = item['image']
        conversations = item['conversations']
        

        for i in range(0, len(conversations), 2):
            round_convo = {
                "id": id,
                "image": image,
                "conversations": conversations[i:i+2]
            }
            new_data.append(round_convo)


    with open(output_file, 'w') as file:
        json.dump(new_data, file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process conversations in a JSON file.')
    parser.add_argument('--input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, help='Path to the output JSON file')
    args = parser.parse_args()
    
    process_file(args.input_file, args.output_file)
