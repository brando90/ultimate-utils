

# def draw_nx(g, labels=None):
#     import matplotlib.pyplot as plt
#     if labels is not None:
#         g = nx.relabel_nodes(g, labels)
#     pos = nx.kamada_kawai_layout(g)
#     nx.draw(g, pos, with_labels=True)
#     plt.show()
#
# def draw_nx_attributes_as_labels(g, attribute):
#     # import pylab
#     import matplotlib.pyplot as plt
#     import networkx as nx
#     labels = nx.get_node_attributes(g, attribute)
#     pos = nx.kamada_kawai_layout(g)
#     nx.draw(g, pos, labels=labels, with_labels=True)
#     # nx.draw(g, labels=labels)
#     # pylab.show()
#     plt.show()
#
# def draw_nx_with_pygraphviz(g, path2file=None, save_file=False):
#     attribute_name = None
#     draw_nx_with_pygraphviz_attribtes_as_labels(g, attribute_name, path2file, save_file)
#
# def draw_nx_with_pygraphviz_attribtes_as_labels(g, attribute_name, path2file=None, save_file=False):
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg
#
#     # https://stackoverflow.com/questions/15345192/draw-more-information-on-graph-nodes-using-pygraphviz
#     # https://stackoverflow.com/a/67442702/1601580
#
#     if path2file is None:
#         path2file = './example.png'
#         path2file = Path(path2file).expanduser()
#         save_file = True
#     if type(path2file) == str:
#         path2file = Path(path2file).expanduser()
#         save_file = True
#
#     print(f'\n{g.is_directed()=}')
#     g = nx.nx_agraph.to_agraph(g)
#     if attribute_name is not None:
#         print(f'{g=}')
#         # to label in pygrapviz make sure to have the AGraph obj have the label attribute set on the nodes
#         g = str(g)
#         g = g.replace(attribute_name, 'label')
#         print(g)
#         # g = pgv.AGraph(g)
#         g = pgv.AGraph(g)
#     g.layout()
#     g.draw(path2file)
#
#     # https://stackoverflow.com/questions/20597088/display-a-png-image-from-python-on-mint-15-linux
#     img = mpimg.imread(path2file)
#     plt.imshow(img)
#     plt.show()
#
#     # remove file https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder
#     if not save_file:
#         path2file.unlink()

# tests

def test1():
    pass

if __name__ == '__main__':
    test1()
    print("Done\a")