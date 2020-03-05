import os

from PIL import Image, ImageDraw


class TreeDrawing:
    if not os.path.isdir(f'../../tmp'):
        os.mkdir(f'../../tmp')
    if not os.path.isdir(f'../../tmp/HistoryFiles'):
        os.mkdir(f'../../tmp/HistoryFiles')
    if not os.path.isdir(f'../../tmp/HistoryFiles/Trees'):
        os.mkdir(f'../../tmp/HistoryFiles/Trees')
    if not os.path.isdir(f'../../tmp/HistoryFiles/Trees/pop_individuals'):
        os.mkdir(f'../../tmp/HistoryFiles/Trees/pop_individuals')
    if not os.path.isdir(f'../../tmp/HistoryFiles/Trees/mutation'):
        os.mkdir(f'../../tmp/HistoryFiles/Trees/mutation')
    if not os.path.isdir(f'../../tmp/HistoryFiles/Trees/crossover'):
        os.mkdir(f'../../tmp/HistoryFiles/Trees/crossover')

    def _getwidth(self, node):
        if not node.nodes_from:
            return 1
        else:
            result = 0
            for i in range(0, len(node.nodes_from)):
                result += self._getwidth(node.nodes_from[i])
            return result

    def _draw_node(self, node, draw, x, y):
        if node.nodes_from:
            allwidth = 0
            for c in node.nodes_from:
                allwidth += self._getwidth(c) * 100
            left = x - allwidth / 2
            # draw the function name
            draw.text((x - 10, y - 10), node.eval_strategy.model.__class__.__name__, (0, 0, 0))

            # draw the children
            for c in node.nodes_from:
                wide = self._getwidth(c) * 100
                draw.line((x, y, left + wide / 2, y + 100), fill=(255, 0, 0))
                self._draw_node(c, draw, left + wide / 2, y + 100)
                left = left + wide
        else:
            draw.text((x - 5, y), str(node.eval_strategy.model.__class__.__name__), (0, 0, 0))

    def draw_branch(self, node, jpeg="tree.png"):

        w = self._getwidth(node) * 100
        if not node.nodes_from:
            h = 100 + 120
        else:
            h = node.get_depth_down() * 100 + 120

        img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        self._draw_node(node, draw, w / 2, 20)
        img.save(f'../../tmp/HistoryFiles/Trees/{jpeg}', 'PNG')
