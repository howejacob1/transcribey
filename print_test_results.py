#!/usr/bin/env python3

import ast
import sys

def print_vcon_info(vcon_dict):
    """Print formatted information for a single vcon"""
    
    # Extract UUID (from _id field)
    uuid = vcon_dict.get('_id', 'N/A')
    
    # Extract languages from analysis[0]["body"]["languages"]
    languages = 'N/A'
    if 'analysis' in vcon_dict and len(vcon_dict['analysis']) > 0:
        analysis_body = vcon_dict['analysis'][0].get('body', {})
        if 'languages' in analysis_body:
            languages = ', '.join(analysis_body['languages'])
    
    # Extract filename from dialog[0]["filename"] and remove /home/bantaim/conserver/
    filename = 'N/A'
    if 'dialog' in vcon_dict and len(vcon_dict['dialog']) > 0:
        full_filename = vcon_dict['dialog'][0].get('filename', '')
        if full_filename:
            # Remove the prefix /home/bantaim/conserver/
            prefix = '/home/bantaim/conserver/'
            if full_filename.startswith(prefix):
                filename = full_filename[len(prefix):]
            else:
                filename = full_filename
    
    # Extract text from dialog[0]["transcript"]
    text = 'N/A'
    if 'dialog' in vcon_dict and len(vcon_dict['dialog']) > 0:
        transcript = vcon_dict['dialog'][0].get('transcript', {})
        if isinstance(transcript, dict) and 'text' in transcript:
            text = transcript['text']
    
    # Print formatted output
    print("--------")
    print(f"ID: {uuid}")
    print(f"languages: {languages}")
    print(f"filename: {filename}")
    print(f"text: {text}")

def main():
    try:
        # Read the entire file content
        with open('test_recordings_output.txt', 'r') as f:
            content = f.read().strip()
        
        # Parse the content as a Python literal (list of dictionaries)
        vcon_list = ast.literal_eval(content)
        
        # Print info for each vcon
        for vcon in vcon_list:
            print_vcon_info(vcon)
        
        print("--------")
        print(f"Total records processed: {len(vcon_list)}")
        
    except FileNotFoundError:
        print("Error: test_recordings_output.txt not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 