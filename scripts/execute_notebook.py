from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from nbconvert import HTMLExporter
root=Path(__file__).resolve().parents[1]
path=root/'notebooks/01_medical_costs.ipynb'
b=nbformat.read(path,as_version=4)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(b,timeout=300,kernel_manager=manager,resources={'metadata':{'path':str(root)}}).execute()
nbformat.validate(b);nbformat.write(b,path)
html,_=HTMLExporter(template_name='classic').from_notebook_node(b)
(root/'reports/eda.html').write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n')
print('Executed notebook and exported reports/eda.html')
