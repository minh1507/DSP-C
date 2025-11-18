# Tạo môi trường ảo
python -m venv vuln_env

# Kích hoạt môi trường ảo trên Windows PowerShell
vuln_env\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt


# trong Windows PowerShell
git clone https://github.com/tree-sitter/tree-sitter-c
git clone https://github.com/tree-sitter/tree-sitter-cpp
git clone https://github.com/tree-sitter/tree-sitter-java

# build library
python -c "from tree_sitter import Language; Language.build_library('build/my-languages.so', ['tree-sitter-c','tree-sitter-cpp','tree-sitter-java'])"

python run_pipeline.py

/dataset
https://zenodo.org/records/4734050