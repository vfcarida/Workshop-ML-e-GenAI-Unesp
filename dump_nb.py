import json

try:
    with open('ViniCarida_Workshop_Unesp_RioPRetoi.ipynb', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    
    with open('workshop_dump.py', 'w', encoding='utf-8') as f:
        for i, c in enumerate(code_cells):
            src = ''.join(c.get('source', []))
            f.write(f"# CELL {i}\n")
            f.write(src)
            f.write("\n\n")
            
    print("Dumped to workshop_dump.py")
except Exception as e:
    print(f"Error: {e}")
