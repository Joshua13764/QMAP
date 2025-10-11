# demo_nodegraphqt_healed_execute_minimal_fixed.py
# Minimal NodeGraphQt demo — uses documented NodeGraph signals (port_connected / port_disconnected)
from dataclasses import dataclass, field
from Qt import QtWidgets
from NodeGraphQt import NodeGraph, BaseNode
import sys

# --- minimal connection types ---------------------------------------------
@dataclass(frozen=True)
class ConnectionRef:
    node: "HealedBaseNode"
    port_name: str

    def __post_init__(self):
        cls = globals().get("HealedBaseNode", None)
        if cls is None or not isinstance(self.node, cls):
            raise TypeError("ConnectionRef.node must be an instance of HealedBaseNode")


@dataclass
class PortConnections:
    port_name: str
    links: list = field(default_factory=list)
    def add(self, ref: ConnectionRef):
        if ref not in self.links:
            self.links.append(ref)
    def remove(self, ref: ConnectionRef):
        if ref in self.links:
            self.links.remove(ref)

class ConnectionMap:
    def __init__(self):
        self._map = {}
    def add(self, port_name, ref: ConnectionRef):
        if port_name not in self._map:
            self._map[port_name] = PortConnections(port_name)
        self._map[port_name].add(ref)
    def remove(self, port_name, ref: ConnectionRef):
        g = self._map.get(port_name)
        if g is None:
            return
        g.remove(ref)
        if not g.links:
            del self._map[port_name]
    def items(self):
        return self._map.items()
    def is_stub(self):
        return not bool(self._map)

# --- healed base node with execute/receive API -----------------------------
class HealedBaseNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.incoming = ConnectionMap()
        self.outgoing = ConnectionMap()
        self._received = {}

    # connection bookkeeping
    def add_incoming(self, src_node, src_port_name, dst_port_name):
        self.incoming.add(dst_port_name, ConnectionRef(src_node, src_port_name))
    def remove_incoming(self, src_node, src_port_name, dst_port_name):
        self.incoming.remove(dst_port_name, ConnectionRef(src_node, src_port_name))
    def add_outgoing(self, dst_node, dst_port_name, src_port_name):
        self.outgoing.add(src_port_name, ConnectionRef(dst_node, dst_port_name))
    def remove_outgoing(self, dst_node, dst_port_name, src_port_name):
        self.outgoing.remove(src_port_name, ConnectionRef(dst_node, dst_port_name))

    # evaluation API (override in subclasses)
    def process(self):
        return None

    def on_input(self, value, port_name):
        self._received[port_name] = value
        return value

    # helpers
    def outgoing_is_stub(self):
        return self.outgoing.is_stub()
    def outgoing_items(self):
        return self.outgoing.items()
    def incoming_items(self):
        return self.incoming.items()

    # propagation methods
    def execute_propagate(self, initial_value=None, visited=None):
        if visited is None:
            visited = set()

        value = initial_value if initial_value is not None else self.process()
        if value is None:
            return

        for out_port_name, port_conns in self.outgoing_items():
            for ref in port_conns.links:
                key = (id(self), out_port_name, id(ref.node), ref.port_name, value)
                if key in visited:
                    continue
                visited.add(key)
                ref.node.receive_from(self, out_port_name, ref.port_name, value, visited)

    def receive_from(self, src_node, src_port_name, dst_port_name, value, visited):
        result = self.on_input(value, dst_port_name)
        if result is not None:
            self.execute_propagate(initial_value=result, visited=visited)

# --- example nodes --------------------------------------------------------
class InputNode(HealedBaseNode):
    __identifier__ = 'com.example'
    NODE_NAME = 'Input Node'
    def __init__(self):
        super().__init__()
        self.add_output('value')
        self.value = 1.0
    def process(self):
        return self.value

class SquareNode(HealedBaseNode):
    __identifier__ = 'com.example'
    NODE_NAME = 'Square Node'
    def __init__(self):
        super().__init__()
        self.add_input('in'); self.add_output('out')
        self.last_output = None
    def on_input(self, val, port_name):
        self.last_output = val * val
        print(f"[SquareNode] {val} -> {self.last_output}")
        return self.last_output
    def process(self):
        return self.last_output

class PrintNode(HealedBaseNode):
    __identifier__ = 'com.example'
    NODE_NAME = 'Print Node'
    def __init__(self):
        super().__init__()
        self.add_input('in')
    def on_input(self, val, port_name):
        print(f"[PrintNode] value = {val}")
        return None

# --- connection manager (uses documented signals) --------------------------
class ConnectionManager:
    def __init__(self, graph):
        self._graph = graph
        # NodeGraphQt docs expose port_connected / port_disconnected signals
        graph.port_connected.connect(self._on_connected)
        graph.port_disconnected.connect(self._on_disconnected)

    def _pick_ports(self, a, b):
        # documented Port API exposes is_input()
        if a.is_input():
            return a, b
        if b.is_input():
            return b, a
        # otherwise check node inputs() map and choose accordingly
        if a.name() in a.node().inputs():
            return a, b
        return b, a

    def _on_connected(self, a, b):
        in_port, out_port = self._pick_ports(a, b)
        src, dst = out_port.node(), in_port.node()
        src.add_outgoing(dst, in_port.name(), out_port.name())
        dst.add_incoming(src, out_port.name(), in_port.name())

    def _on_disconnected(self, a, b):
        in_port, out_port = self._pick_ports(a, b)
        src, dst = out_port.node(), in_port.node()
        src.remove_outgoing(dst, in_port.name(), out_port.name())
        dst.remove_incoming(src, out_port.name(), in_port.name())

    def update_from_graph(self):
        for p in self._graph.get_connected_pipes():
            ip, op = p.input_port(), p.output_port()
            self._on_connected(ip, op)

# --- evaluator (single-line list comprehension) ---------------------------
def evaluate_from_graph(graph):
    [n.execute_propagate() for n in graph.all_nodes() if not n.outgoing_is_stub()]

# --- app / UI --------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    graph = NodeGraph()
    graph.register_node(InputNode)
    graph.register_node(SquareNode)
    graph.register_node(PrintNode)

    conn_mgr = ConnectionManager(graph)

    widget = graph.widget
    widget.setWindowTitle('NodeGraphQt — minimal healed execute demo (fixed)')
    widget.resize(900, 600)
    widget.show()

    ik = f"{InputNode.__identifier__}.{InputNode.__name__}"
    sk = f"{SquareNode.__identifier__}.{SquareNode.__name__}"
    pk = f"{PrintNode.__identifier__}.{PrintNode.__name__}"

    n_in = graph.create_node(ik, name='Input', pos=[50, 100])
    n_sq = graph.create_node(sk, name='Square', pos=[300, 100])
    n_pr = graph.create_node(pk, name='Printer', pos=[550, 100])

    n_in.value = 3.0

    out = n_in.output(0)
    out.connect_to(n_sq.inputs()['in'])
    out.connect_to(n_pr.inputs()['in'])

    conn_mgr.update_from_graph()

    btn_eval = QtWidgets.QPushButton('Eval', widget); btn_eval.setGeometry(10,10,80,28); btn_eval.show()
    btn_eval.clicked.connect(lambda: evaluate_from_graph(graph))

    btn_set5 = QtWidgets.QPushButton('Set=5', widget); btn_set5.setGeometry(100,10,80,28); btn_set5.show()
    btn_set5.clicked.connect(lambda: (setattr(n_in,'value',5.0), evaluate_from_graph(graph)))

    btn_dump = QtWidgets.QPushButton('Dump', widget); btn_dump.setGeometry(190,10,80,28); btn_dump.show()
    def dump():
        for n in (n_in, n_sq, n_pr):
            print(f"--- {n.NODE_NAME} ---")
            print("incoming:", {k: [(r.node.NODE_NAME, r.port_name) for r in v.links] for k,v in n.incoming_items()})
            print("outgoing:", {k: [(r.node.NODE_NAME, r.port_name) for r in v.links] for k,v in n.outgoing_items()})
    btn_dump.clicked.connect(dump)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
