# demo_nodegraphqt_fixed.py
# Works with PySide6 / PyQt6 / PyQt5 / PySide2 (whichever is installed)
from Qt import QtWidgets
from NodeGraphQt import NodeGraph, BaseNode
import sys

class MySimpleNode(BaseNode):
    __identifier__ = 'com.example'
    NODE_NAME = 'Simple Node'
    def __init__(self):
        super(MySimpleNode, self).__init__()
        self.add_input('in')
        self.add_output('out')

def main():
    app = QtWidgets.QApplication(sys.argv)

    graph = NodeGraph()

    # register node before creation
    graph.register_node(MySimpleNode)

    # print registered nodes (debug)
    try:
        print('registered_nodes():', graph.registered_nodes())
    except Exception:
        print('fallback registered keys:', list(graph._node_factory.nodes.keys()))

    widget = graph.widget
    widget.setWindowTitle('NodeGraphQt — Minimal Demo (fixed)')
    widget.resize(800, 600)
    widget.show()

    # build node key from class to avoid mismatches
    node_key = f"{MySimpleNode.__identifier__}.{MySimpleNode.__name__}"
    print('creating node with key:', node_key)

    node_a = graph.create_node(node_key, name='Node A', pos=[50, 100])
    node_b = graph.create_node(node_key, name='Node B', pos=[350, 100])

    out_port = node_a.output(0)
    in_port = node_b.inputs()['in']
    out_port.connect_to(in_port)

    # optional: focus
    try:
        widget.centerOn(200, 150)
    except Exception:
        pass

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
