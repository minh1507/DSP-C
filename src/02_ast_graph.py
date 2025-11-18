from tree_sitter import Language, Parser
import networkx as nx

# build my-languages.so trước 1 lần: tree-sitter-c, tree-sitter-cpp, tree-sitter-java
C_LANGUAGE = Language('build/my-languages.so', 'c')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')

def get_ast_graph(code:str, lang:str):
    parser = Parser()
    if lang=="c":
        parser.set_language(C_LANGUAGE)
    elif lang=="cpp":
        parser.set_language(CPP_LANGUAGE)
    elif lang=="java":
        parser.set_language(JAVA_LANGUAGE)
    else:
        raise ValueError(f"Language {lang} not supported")

    tree = parser.parse(bytes(code, 'utf8'))
    G = nx.DiGraph()

    def traverse(node, parent=None):
        node_id = id(node)
        G.add_node(node_id, type=node.type)
        if parent is not None:
            G.add_edge(parent, node_id)
        for c in node.children:
            traverse(c, node_id)

    traverse(tree.root_node)
    return G
