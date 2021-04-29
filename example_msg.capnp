@0xc16a95360f09f2fc;

struct Dag {
  using File = Text;
  using DepIndex = UInt16;
  using NodeIndex = UInt32;

  # The first file is always the current file, so beware for loops
  dependencies @0 :List(File);

  nodes @1 :List(Node);
  proofSteps @2 :List(ProofStep);

  struct NodeRef {
    depIndex @0 :DepIndex;
    nodeIndex @1 :NodeIndex;
  }

  struct Node {
    union {
      leaf @0 :LeafKind;
      node :group {
        kind @1 :NodeKind;
        children @2 :List(NodeRef);
      }
    }
  }
}