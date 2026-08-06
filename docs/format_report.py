import re

def format_report(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    formatted_lines = []
    
    for line in lines:
        original = line.strip()
        
        # Match Chapter headings (e.g., "Chapter 1")
        if re.match(r'^Chapter\s+\d+$', original, re.IGNORECASE):
            formatted_lines.append(f"\n# {original}\n")
            continue
            
        # Match major sections (e.g., "1.1 Background of the project")
        if re.match(r'^\d+\.\d+\s+[A-Za-z]', original):
            formatted_lines.append(f"\n## {original}\n")
            continue
            
        # Match sub-sections (e.g., "2.3.1 Problem Statement")
        if re.match(r'^\d+\.\d+\.\d+\s+[A-Za-z]', original):
            formatted_lines.append(f"\n### {original}\n")
            continue
            
        # Match major standalone titles
        if original in ["Abstract", "Contents", "List of Figures", "List of Tables", "References"]:
            formatted_lines.append(f"\n# {original}\n")
            continue
            
        # Figure and Table captions
        if re.match(r'^Figure\s+\d+\.\d+:', original) or re.match(r'^Table\s+\d+\.\d+:', original):
            formatted_lines.append(f"\n*{original}*\n")
            continue
            
        # Normal text (add paragraph breaks for readability)
        if original:
            formatted_lines.append(f"{original}\n\n")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(formatted_lines)
        
    print(f"Successfully formatted {input_file} -> {output_file}")

if __name__ == "__main__":
    format_report('skyra_data/reportresume.txt', 'TruthSeeker_Report.md')
