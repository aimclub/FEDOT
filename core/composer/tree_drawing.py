from PIL import Image, ImageDraw
import os

class Tree_Drawing:
    def __init__(self, secondary_node_type, primary_node_type):
        self.secondary_node_type = str(secondary_node_type)
        self.primary_node_type = str(primary_node_type)

    def getwidth(self, node):
        if str(type(node)) == self.primary_node_type:
            return 1
        elif str(type(node)) == self.secondary_node_type:
            result = 0
            for i in range(0, len(node.nodes_to)):
                result += self.getwidth(node.nodes_to[i])
            return result

    def drawnode(self, node, draw, x, y):
        if str(type(node)) == self.secondary_node_type:
            allwidth = 0
            for c in node.nodes_to:
                allwidth += self.getwidth(c) * 100
            left = x - allwidth / 2
            # draw the function name
            draw.text((x - 10, y - 10), node.eval_strategy.model.__class__.__name__, (0, 0, 0))

            # draw the children
            for c in node.nodes_to:
                wide = self.getwidth(c) * 100
                draw.line((x, y, left + wide / 2, y + 100), fill=(255, 0, 0))
                self.drawnode(c, draw, left + wide / 2, y + 100)
                left = left + wide
        elif str(type(node)) == self.primary_node_type:
            draw.text((x - 5, y), str(node.eval_strategy.model.__class__.__name__), (0, 0, 0))

    def draw_branch(self, node, jpeg="tree.png"):
        if not os.path.isdir(f'HistoryFiles'):
            os.mkdir(f'HistoryFiles')
        if not os.path.isdir(f'HistoryFiles/Trees'):
            os.mkdir(f'HistoryFiles/Trees')

        w = self.getwidth(node) * 100
        if not str(type(node)) == self.secondary_node_type:
            h = 100 + 120
        else:
            h = node.get_depth_down() * 100 + 120

        img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        self.drawnode(node, draw, w / 2, 20)
        img.save(f'HistoryFiles/Trees/{jpeg}', 'PNG')
